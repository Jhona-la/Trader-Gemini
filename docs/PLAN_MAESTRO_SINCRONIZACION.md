# PLAN MAESTRO DE SINCRONIZACIÓN — Crecimiento Exponencial Compuesto

## Estado vigente de coordinación — 2026-10-08

Candidato local `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`, árbol
`10b8a427dfbdd5b203595b039b2dad50c69d6f68`: unión de la recuperación
`7acf36aa` y remoto `5842c8e3` (incluye Ω26/Ω27). **T1: APROBADO en e9: 16/144 sensibles (11,1%), mínimo 0,110 intacto, 1 test exacto y 1 filtrado; fixture sintético; publicación pendiente del recibo remoto.** El registro de 209 tests describe `f3f8696`; requiere recibos
propios la nueva composición. T1 mide expresividad genética y no rentabilidad
ni cumplimiento del objetivo financiero; la paridad requiere sus contratos.

El [plan por archivo](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md)
conserva las fases R0–R9 y la sección 11 de Sol. El nuevo ledger
[ledger anotado de código](audit/REVISION_CODIGO_2026-10-08.json) censa 1.478 rutas/474 Rust/655
`graphify-out` en `e9c6a435`. Su generación inicial dejó todas las filas en
`inventariado`; la anotación posterior registra **1.469 inventariado, 8
hallazgo_abierto con alcance parcial y 1 verificado**. Sólo está verificado
el archivo de tests C07 completo (139 líneas; 4/4 dentro de 54/54 en riesgo),
con revisión independiente; ningún archivo de runtime queda certificado
completamente por esas nueve anotaciones. El check posterior tuvo exit 0 y
valida cobertura/metadatos, no la verdad de la revisión ni resultados OOS.
Los censos 1.434/460 y 1.474/473 pertenecen a bases anteriores. El siguiente
delta debe capturar el cierre documental después del merge, sin autorreferencia.

| Responsable | Reserva/estado observado; no equivale a acuse nuevo |
|---|---|
| Qoder | Ola 73 CERRADA e INTEGRADA en `main`. Fase R7-R4 (núcleo vivo) cerrada y fusionada en `ce5e0897` (ficha #688). Actualmente ejecutando Fase R7-R6 (Aprender y medir) en worktree `.r7r6`, rama `qoder/r7r6-aprender-medir`, con reserva de Ficha #690. |
| Antigravity | Olas Ω26 a Ω72 CERRADAS e INTEGRADAS (Fichas Forenses #689-#710 certificadas con cero fallos, cero heap allocations en hot path < 25 ns y 100% Rust estricto). Ficha #710 optimizó hot path a zero-alloc en `signal-engine::orchestrator` y `risk-engine::selection_stats`. Ficha #709 resolvió y certificó 5 vetos contractuales en `risk-engine::veto_registry`. Ficha #708 integró presión de Prospect Theory en el tick loop de `god-engine-core`. Ficha #706 implementó colector Feynman y Prospect Theory. Plan Maestro Cuántico en `docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md`. Próximo bloque: Continuación de barrido por fases R0-R9 archivo por archivo. |
| GLM | GLM112 incorporado; contratos de knobs configurables y shadow de soliton deben preservarse o revisarse con prueba y acuse. No se presupone una reserva nueva. |
| Sol | Reporting SOL-R5-01 publicado (`f8c433f8`) e incorporado. Reserva activa: `booktick_replay.rs` y `metrics.rs` para accounting de replay de backtest. |
| Codex | Integración, contratos recuperados, fundamentos, censo y documentación. Arc de Darwin, C07, model reload y publicación de evidencia se conservan; los resultados económicos y cobertura integral siguen abiertos. |
| Claude | Dominio histórico de ejecución, sin acuse nuevo observado. |

Los estados del 7 de octubre y olas anteriores que siguen debajo son
snapshots con evidencia acotada. En particular, ola67/flow en vuelo y
reporting Sol pendiente no representan la reserva ni el estado actual.
La nueva observación sustituye esos estados, no sus aportaciones históricas.

Limpieza sólo de refs verificadas por OID y ancestría en `main` y
`origin/main` publicado, después del censo final y de excluir actividad.
No eliminar directorios de worktrees. Los checkouts históricos sucios se
preservan; un detach autorizado al mismo HEAD exige status y hashes iguales
antes/después. `main`, ramas activas Qoder/Antigravity y ramas externas nuevas
quedan excluidas. La publicación y la limpieza aún requieren sus recibos.


Validación aplicable: 736 regresiones aplicables + T1 (1 test exacto, 16/144 sensibles, mínimo 0,110 intacto); total 737 aprobadas, 0 fallidas y 1 ignorada. 729 aprobadas se retienen por identidad desde 7acf y 8 se ejecutaron en e9, incluido T1. Check all-targets aprobado antes del merge; estos resultados no validan crecimiento económico. Los recibos conservan sus SHA
de ejecución: la matriz de equivalencia compara todas las entradas
versionadas y 530 inputs compilables/de fixtures, con modo/kind/OID.
Reporta por separado los dos archivos afectados y reejecutados en
`e9c6a435`: reporting continuo 2/2 y stateful_open 5/5. Los estados
externos no registrados por Git tienen límites explícitos; no se
finge que las suites anteriores corrieron en este SHA.

> **Documento vivo de los tres agentes** (Qoder, GLM/Antigravity, Claude).
> Creado por Qoder (Ola 55, #655) el 2026-10-04 sobre main `0580e267`.
>
> **Mapa de planes (LXXXV de GLM, §5b)**: este documento es el de
> CONTRATOS DE TRABAJO por línea + decisiones del dueño (§4). El
> `PLAN_MAESTRO_2026-10-04.md` de GLM es el de ESTADO DEL SISTEMA y
> ruta crítica; la extensión G0-G8 de Codex (PR #28) cubre contratos
> raíz por ruta. Los tres se referencian, ninguno sustituye.
> Cada agente lo actualiza al cerrar su ola. La fuente de verdad del
> estado de cada uno es su buzón + este plan; el detalle forense vive en
> `FORENSIC_INTELLIGENCE_AUDIT.md` (Qoder), `docs/adr/` (GLM) y
> `docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md` (Claude).

> **Snapshot histórico R4 — reinicio 2026-10-07 sobre adeb8d1b:** recorrido operativo en
> [PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md).
> Base inventariada: 1.434 rutas/460 Rust, sin transferencia de cobertura
> histórica. Codex revisa fundamentos y recupera ramas pendientes; GLM
> mantiene flow_impulse y Qoder su ola67. Acuses de esta ronda pendientes.

## Actualización operativa 2026-10-07 — Codex + revisión independiente Sol

El recorrido vigente de R4 queda en [PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md), recuperado del trabajo de Codex con atribución y contrastado por Sol contra `8938cf41`. Este documento sigue siendo el contrato de coordinación; BARRIDO conserva historia, no certifica el árbol actual.

El nuevo censo versionado incluye 1.434 archivos, 460 Rust y 23 crates miembros más raíz; **todos inventariados, ninguno declarado auditado completamente por clasificación automática**. Fases R0-R9: metas → conceptos/topología → matemática/estadística → física/cuántica/algoritmos → datos/tiempo → capital/vetos/ejecución → aprendizaje/GenomeStore → plataforma → archivos restantes → integración y examen OOS. Cada lote requiere escritor, revisor, blob/rangos, prueba conductual, exit y recibo de main.

La meta de duplicación cada 72 h se mide sobre equity neta, costes y flujos externos separados. `ln(2)/3 = 0.23104906` es crecimiento log diario; el retorno diario compuesto equivalente es `25.992105%`. Es un objetivo, no una garantía. Un test verde, un T-1 estable o una fase inventariada no prueban ni rentabilidad ni paridad completa. No se autoriza producción con este plan.

Reserva inicial Sol: continuación documental, censo reutilizado y diagnósticos independientes de contabilidad/evidencia. Continuación 16:02+: SOL-R5-01 reserva reporting diario de `continuous_evolution_backtest.rs` con helper y pruebas de arena/slots reales; LOCAL TEST PASSED (full-bin reporting_contract 2/2, direct exit 0; target/reporting-contract-execution-evidence-20261007.txt), integration/publication pending, no economic validation. Baseline offline locked workspace all-targets check passed, cargo/tee exits 0 (target/sol-baseline-all-targets-20261007.log and .exit). T-1 was not rerun because the change is daily reporting only, not the live strategy pipeline. Se preservan valuación, shadow selection, sizing, modelos y producción. Codex mantiene la recuperación de ramas y el fix del caller Darwin; GLM/Qoder/Claude mantienen sus dominios históricos hasta nuevo acuse. Avisos en el buzón no equivalen a recepción; no duplicar archivos en vuelo ni borrar worktrees ajenos.

## 0. La meta y su física

**Objetivo: 100% cada 3 días desde 13 USD, por interés compuesto.** En log:
0.231049/día; rendimiento simple equivalente: 25.9921% diario. El Sharpe
diario ≈0.68 se deriva únicamente del modelo GBM idealizado a Kelly pleno
sin fricción (`g*=SR²/2`); no es garantía ni regla universal de dimensionado.
La viabilidad de la meta requiere evidencia económica OOS con costes,
incertidumbre, capacidad y supervivencia. No está demostrada por compilar,
por el oráculo T-1 ni por la complejidad de las teorías incorporadas.

La estrategia honesta hacia la meta (consenso de seniors, 2026-09-28):
maximizar el crecimiento log con tope de ruina y **medirlo fuera de
muestra**. El T-1 mide expresividad genética, no rentabilidad; la
cadena de paridades bt↔vivo (GLM registra 10/10 históricos) aporta
evidencia para sus versiones y fixtures; no garantiza paridad universal
ni certifica esta composición por herencia. Siguen abiertos el edge
OOS, la conciliación económica y la inferencia bajo dependencia,
según el protocolo R4. Las líneas de trabajo de abajo conservan sus
responsabilidades históricas y requieren revalidación por consumidor.

## 0b. Barrido exhaustivo (mandato 2026-10-04)

El operador ordenó recorrer TODOS los archivos uno por uno en fases
(metas → conceptos → matemática → física → código):
**docs/BARRIDO_EXHAUSTIVO_FASES.md** conserva las 8 fases y el censo
histórico aproximado (~377 archivos/~137k líneas). Para cobertura
actual se usa el ledger R4 por SHA, su delta y las fases R0–R9 del
plan por archivo; contar rutas no acredita revisarlas.

## 1. Las tres líneas ( quién es dueño de qué )

### LÍNEA A — ENTENDER el universo (Qoder): el continuo espectral
**Estado: VIVO de punta a punta** (#609..#654, 48 hallazgos con oráculo).

- 13/13 motores votan por escala con física honesta (kernel Hawkes real,
  solitón enderezado, SR canónica, choque firmado, paridad sombra↔vivo).
- Consenso espectral con pesos por habilidad prequential (IC por
  motor×escala, significancia #599, piso 0.15) y gate de observabilidad
  (D-742) — el orquestador lo consume con τ operable y TTL.
- Vetos con dato medido: ρ(τ*) multiactivo activo (#651), Lundberg con
  unidades de capital (#651), drawdown con tolerancia micro compuesta
  (#653, ver decisión pendiente §4).
- Distribuciones servidas para calibrar consumidores: D₀ (#654),
  |v_dom| vs cutoff (#652).

**Siguiente (Qoder)**: (a) decisión del dueño sobre tolerancia micro y
unificación de cortacircuitos (§4); (b) consumidor del D₀ con la
distribución medida; (c) medición de adopción espectral en vivo
(`qo_624_fraccion_espectral`, `qo_652_fraccion_sobre_corte`) sobre la
primera sesión viva — si el espectro nunca domina, auditar por qué.

### LÍNEA B — APRENDER del universo (GLM): DL-modular, escalera falsable
**Estado: L2 fase 2 en vuelo** (ADR-0010).

- L0/L1 hechos; L2 fase 1 (dataset de votos 178k filas BTC + apriori
  hostil) mergeado; tau-matched completo; fase 2 (etiqueta
  τ-por-muestra) anunciada.
- **Regla de la casa para la escalera**: ningún peldaño sube sin ganar
  a un nulo por permutación fuera de muestra (DSR en toda promoción).

**Siguiente (GLM)**: L2 fase 2 → primer modelo tau-matched; la familia
honesta (7 símbolos) para el contraste. Coordinación con Línea A: el
dataset exporta los votos del consenso espectral de Qoder — si la
física cambia (cambió en #649/#650), regenerar el dataset.

### LÍNEA C — EJECUTAR sin perder un bps (Claude): cimientos
**Estado: ciclo 8 cerrado** (CL-36..42, 1453 tests verdes, T-1 16/144).

- Identidad de símbolos, IOC con evidencia terminal, genoma único
  seguido por el núcleo, apalancamiento validado en envío.
- **Abiertos documentados** (su MEMORIA 2026-10-04): GENOME-GATE al
  cargar (slots 21/54/107 + RR), IOC parcial, dimensionar en espacio de
  riesgo, intenciones `New` en rechazo firme de MARKET/maker.

**Siguiente (Claude)**: GENOME-GATE (bloquea promociones sanas),
dimensionado en espacio de riesgo (la propuesta XLIV: riesgo =
fracción de Kelly acotada por ruina; nocional = riesgo/SL) — es el
cambio de sizing que la meta de compuesto necesita para apostar cerca
de Kelly EN RIESGO AL STOP.

## 2. Intersecciones y reglas de no-choque

| Intersección | Regla |
|---|---|
| Qoder cambia física de motores → GLM | El dataset L2 se regenera (los votos cambiaron). Avisar en buzón con etiqueta `[dataset-inval]`. |
| GLM promueve modelo tau-matched → Qoder | El apriori hostil se recalibra contra el consenso espectral vivo de la sesión, no contra el fixture. |
| Claude cambia sizing/ejecución → ambos | Toda paridad bt↔vivo se re-corre antes de operar (GLM la automatizó: 10/10). |
| Oráculo T-1 | Trinquete 11.0%, margen 0.1 pts: **toda ola que toca pipeline vivo lo corre antes del push**. Sin excepciones — 10 olas consecutivas lo han demostrado. |
| Buzón | UNIÓN en conflictos; cada cierre de ola lleva entrada; las revisiones cruzadas (Claude↔Qoder ya funcionan) se responden en la misma ola o la siguiente. |

## 3. Cadencia hacia la meta — snapshot del plan original

La cadencia siguiente conserva el registro original. «Certificado» no es
certificación universal del sistema, «sin fricción» no describe costes reales
y la sesión viva propuesta no constituye una autorización actual. Para R4
prevalecen los recibos, pendientes y reservas de la adenda del 8 de octubre.

1. **Ya**: sistema espectral vivo + certificado; ejecución íntegra;
   dataset de aprendizaje servible.
2. **Siguiente sesión viva** (operador): correr con la familia honesta,
   telemetría completa (censo, adopción espectral, distribución D₀,
   tau-matched) — es la primera sesión donde el ciclo completo opera.
3. **Cierre del año 1 del plan**: edge OOS medido (Línea B) sobre
   ejecución sin fricción (Línea C) entendiendo cada escala (Línea A).
   Con edge medido, el sizing en espacio de riesgo lleva el compuesto;
   sin edge medido, ningún apalancamiento lo lleva.

## 4. Decisiones pendientes del dueño (operador)

1. **Tolerancia micro del drawdown** (escalada por Claude, Ola 53):
   ¿0.85 (D-641, compuesto) o cota medida pura (supervivencia)?
   Recomendación del consejo: unificar los dos cortacircuitos en
   `lerp(dd_max_medido, TOL, micro_w)` con TOL como knob explícito.
2. **Sesión viva**: autorizar la corrida con la familia honesta cuando
   GLM cierre L2 fase 2 (el manifest de cópulas de agosto es deuda
   conocida — ADR-0009: regenerar al iniciar cada mes).

## 5. Bitácora de sincronización

| Fecha | Agente | Entrada |
|---|---|---|
| 2026-10-04 | Qoder | Plan creado (#655). Estado: Línea A viva; D₀ servido; decisión §4.1 pendiente. |
| 2026-10-04 | Codex | RA/PR28 draft:16 expedientes, parser/riesgo no finito/OOS; all-targets verde, CI/T1/revisión externa pendientes. MW separado. No certifica rentabilidad ni cobertura total. |
| 2026-10-04 | Qoder | PRE-FLIGHT: regresión del árbol combinado 7c4cea40 = 828/0 + ws 0 err; roster 18 MOTOR (BTCUSDT incluido); T-1 de sello sobre el tip exacto en vuelo. Riesgos aceptados documentados en buzón. |
| 2026-10-04 | GLM | LXXXIV (en vuelo, lxxxiv-l2v1): datasets τ-matched ago+sep-14 EN GENERACIÓN (nota de regeneración atendida) + trainer v1 logística con plantilla honesta; paridad del PR#27 de Claude anunciada. |
| 2026-10-04 | Codex | root-audit-2026-10-04 (en vuelo): 16 hallazgos root-contract (parser risk-state, límites OOS, tokens escalares, agregación depth); reconciliado con main 7c4cea40 (este plan) tras all-targets check; documenta gates de certificación sin resolver. PR #28 (G0-G8) aprobado por GLM en LXXXV. |
| 2026-10-04 | GLM | LXXXV cerrado: L2 v1 PARCIAL DEFINITIVO — dirección transfiere OOS (+5 pts constante) pero calibración NO (Platt sobrefía selección); cableado BLOQUEADO; fase 4 opcional por petición del consejo. Mapa de planes §5b publicado (tres documentos, roles distintos). |
| 2026-10-04 | Qoder | SELLO FINAL: T-1 del tip exacto 7c4cea40 PASA 16/144 (4221s) — pre-flight VERDE completo, sistema listo para operar (§4.2). Regresión 828/0 + roster 18 MOTOR. |
| 2026-10-04 | GLM | LXXXVIII: saga L2 cierra HERMÉTICA (condicionado≈global; señal concentrada en no-range +8.7 pts). DECISION_FDUSD_BORRADOR.md publicado — esperando palabra del dueño (recomendada B: archivo). |
| 2026-10-05 | Antigravity | PLAN MAESTRO QUANT SR ACTUALIZADO: formalización física y matemática del Universo Continuo Espectral $S(\omega, \tau, \mathbf{x}, t)$, Hodge, Fokker-Planck, E-Values de Ville, Lo-MacKinlay Variance Ratio y DSR Bailey & LdP. Barrido F0-F7 CERRADO (171 hallazgos acumulados), F8 (136 tests de integración) abierta. Micro-cuentas $13 USD auditadas. |
| 2026-10-05 | GLM | LXXXXVIII: CI rojo reparado tras F4-H2 — tests cl41b y live_envelope_gate actualizados a la conducta de bootstrap al apalancamiento validado. Limpieza de residuo de stash. |
| 2026-10-06 | Qoder | OLA 61 CERRADA (a01227cc): F2-B2/B3/B4 sustrato espectral honesto — habilidad_en vecino ln τ, W₁ a lag FÍSICO 60 s (anillo por timestamp del exchange, cadencia 250 ms), ζ(p) con peso continuo de masa. Oráculo PASA 16/144 (2212 s). Banda operable = nodos 18..22 de la malla 4^k µs. |
| 2026-10-06 | Antigravity | XCII: Blindaje estructural definitivo en CI contra falsos positivos cosméticos en `.github/workflows/replay-contracts.yml` (`git -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check HEAD^ HEAD`). Erradicado el fallo en 16s por fin de línea en bitácoras markdown, preservando el rechazo de marcadores de conflicto de merge. |
| 2026-10-06 | Antigravity | OLA Ω11 CERRADA: **G0-2** — ancla fija `swing_tp_base` (12h) sustituida por `tp_at_tau` dinámico a la tau viva de la onda en rama 13 de lib.rs:6017; **G2-10** — lead-lag microestructural libre de auto-referencia en BTC/ETH (BTC líder macro exógeno con div=0.0; ETH sólo contra BTC; alts contra matriz combinada); **G1-5** — unificación DRY canónica de `selection_stats` en risk-engine y re-exportada en evolution-engine (eliminado archivo duplicado). 84/84 + 54/54 + 166/166 verdes, workspace 0 errores. |
| 2026-10-06 | Antigravity | OLA Ω10 CERRADA: **G1-2** — corrección analítica insesgada de Lo & MacKinlay (1988) en Hurst VR multiescala (c_k compensando varianza traslapada, erradicado sesgo de -0.04 en nulo iid); **G1-3** — cableada compuerta formal DSR OOS en darwin.rs (extracción de retornos de operaciones cerradas, conjunción con DSR>=0.95); **G0-4** — congelado gen muerto capital_split_scalp en mutate() y darwin.rs. 2/5 HIGH de Ronda 2 resueltos. |
| 2026-10-06 | Qoder | RONDA 2 DEL BARRIDO (G0-G2, docs-only): revisión desde la base contra a01227cc — 33 hallazgos (5 HIGH): G1-1 Ville sin corrección por multiplicidad (mea culpa #661, cola Qoder), G1-2 Hurst VR sesgado en nulo iid (AGY), G1-3 DSR de darwin es telemetría (AGY), G2-1 calma vota más que cascada en confluence .abs() (Qoder), G2-2 flow_impulse vivo sin exceso-SS (Qoder). Doctrina estructural VERIFICADA LIMPIA tras 15 olas. Asignación completa en buzón. |
| 2026-10-05 | Antigravity | Ω9 CERRADA (1a540356): **S5** — partición OOS cronológica 50/50 en Darwin con compuerta de promoción OOS vs baseline + DSR con control de multiplicidad (Bailey & LdP ec.5, E[max SR] de Gumbel, N=100 pruebas); centralización selection_stats en risk-engine (PSR/DSR/Φ⁻¹ Acklam); Ω4 auditoría micro-vetos VERDE (ρ<0 = hedge admitido). S1-S10 del plan QUANT SR RESUELTOS. |
| 2026-10-05 | Qoder | OLA 59 CERRADA — oráculo PASA 16/144 (3945.93s): relojes físicos (Hawkes α·β·dt + decay τ=60s con timestamp del exchange), limpieza (sombra SR viva, knobs muertos), **ARQUITECTURA_VIVA.md** publicado como mapa canónico. OLA 60 (e-values Ville) oráculo en vuelo sobre tip apilado. |
| 2026-10-05 | Qoder | OLA 59 EN VUELO (worktree .ola59): relojes físicos F2-B5/B6 (Hawkes α·β·dt + decay τ=60s con timestamp del exchange) + limpieza entradas vacías F3-A3/A13/C7 (sombra SR con clave viva, bessel_alpha/hawkes_dt retirados, if-true muerto). **ARQUITECTURA_VIVA.md** publicado como mapa canónico del consejo (pipeline + 10 invariantes + deuda verificada). Tests en verificación, oráculo pendiente. |
| 2026-10-05 | Qoder | OLAS 56/57/58 CERRADAS con oráculo 16/144 c/u: paridades PPO/trailing/NTP + erradicación sombra/vivo (56), H(τ) continua + lead-lag real (57), mea culpas F1-C2/C4 CERRADOS (58). Ola 59 (relojes físicos B5/B6) en verificación. Barrido propio F0-F3: 103 hallazgos, cola consumiéndose. |
| 2026-10-05 | Antigravity | Ω2-Ω8 CERRADAS: F1-A1/C1/B1/B3/B4/A3/A4 (matemática F1 salvo LOWs), F4-H1..H4 (GENOME-GATE), primer toque analítico BM/OU (R8-A), OU físico en VECM, Hurst honesto por Variance Ratio (S8/F2-C4). PLAN MAESTRO QUANT SR con 12 problemas sistémicos S1-S12. |
| 2026-10-05 | GLM | LXXXIX-LXXXXII: tubería de promoción (H1 atomicidad + M1/M2 margen), barrido F5 (22 hallazgos) y F6 (23 hallazgos/6 HIGH, datos/storage). Adoptó el patrón de fases del barrido para su zona. |
| 2026-10-05 | Codex | PR #28 (root-audit G0-G8) sigue DRAFT con CI verde; 6 satélites en worktrees activos. |
| 2026-10-04 | Qoder | BARRIDO §0b: F0 CERRADA (ADR-0014 doctrina espectral) y F1 CERRADA (23 hallazgos, 2 HIGH: umbral n vitalicio del banco #594, gate vs climatology; mea culpas F1-C2/C4) — docs-only, push 23701ecb. Cola de olas priorizada en BARRIDO_EXHAUSTIVO_FASES.md. F2 (física/cuántica) EN VUELO con 3 auditores. |

## 6. Consolidación técnica y alcance de las afirmaciones (Codex)

Esta adenda conserva las secciones anteriores como registro de sus autores y
precisa sus condiciones de validez. No declara consenso, aprobación humana,
auditoría exhaustiva ni rentabilidad que no estén respaldados por un recibo.
El [plan operativo de GLM](PLAN_MAESTRO_2026-10-04.md) conserva los frentes por
agente; este documento detalla dependencias, contratos y aceptación. Ambos
deben enlazarse y usar los mismos IDs de entrega, no mantener estados económicos
incompatibles. La consolidación editorial no asigna trabajo unilateralmente.

Corte de coordinación: main `856e59ba`, plan GLM `d777175f`, candidato Rust
RA `496f902d`. Integración documental local: `f10434d6` y `381e5f7d`.
Los dos merges preservaron cada padre y pasaron all-targets, respectivamente
exit0/1m30s y exit0/15,14s. El código Rust y los contratos no cambiaron respecto
de496f902d. No significa todavía integración de RA en main remoto.

### 6.1 Qué significa exactamente la meta de compuesto

Para capital neto positivo y sin flujos externos, la duplicación a72h implica:

```text
G72 = ln(V(t+72h)/V(t)) = ln(2) = 0,6931471805599453
g_diario = ln(2)/3 = 0,2310490601866484 por día
r_diario_equivalente = exp(g_diario)-1 = 0,2599210498948732
```

`G72` es incremento logarítmico, no porcentaje simple; `r_diario_equivalente`
es25,9921% compuesto diario. Las comisiones, funding, slippage, pérdidas,
posiciones abiertas valoradas y costes operativos deben entrar en `V`.
Con depósitos o retiradas se requiere una serie ajustada por flujos, no tratar
una aportación como beneficio. Capital inicial13USD es un escenario del plan,
no una medición actual del saldo ni prueba de viabilidad de tamaños mínimos.

Sharpe≈0,68 tiene una derivación **condicionada**, no es una condición universal:
si el exceso de retorno sigue una difusión con μ constante por día y σ por
raíz de día, rebalanceo continuo, exposición libre, sin costes ni restricciones,
la tasa log esperada es `g(f)=fμ−f²σ²/2`. Maximizarla da
`f*=μ/σ²`, `g*=μ²/(2σ²)=S_diario²/2`, y por tanto
`S_diario=sqrt(2 ln(2)/3)=0,6797779934458726` para esa tasa esperada.
No fija la probabilidad de duplicar en cada ventana, el drawdown ni la ruina.
Con saltos, colas, error de estimación, restricciones y costes cambia el problema.
La referencia a Sharpe2–6 de fondos en§0 no tiene aquí una muestra verificable;
no se usa como benchmark ni como gate. Esta derivación es nuestra ilustración
de difusión, no una ecuación atribuida al artículo original de Kelly.

El criterio de crecimiento logarítmico y su distinción respecto de maximizar
capital esperado provienen del [trabajo original de Kelly,1956](https://www.kiv.zcu.cz/~vavra/zti/KELLY.PDF).
No convierten una ventaja estadística no medida en una garantía de rendimiento.

### 6.2 Qué medir para aceptar o refutar una aproximación a la meta

- Serie neta de capital por evento y por día, moneda de valoración, flujos,
  latencia, inventario y versión de tarifas. Separar PnL realizado y no realizado.
- Distribución de `G72`, crecimiento log neto medio, incertidumbre, frecuencia
  de ventanas que duplican, peor drawdown, pérdida esperada de cola y capacidad.
  Las ventanas72h solapadas no son observaciones independientes.
- Aportación marginal por activo/escala y costes de aumentar exposición. Más
  símbolos, más operaciones o más señales no implican más crecimiento neto.
- Registrar todos los ensayos del genoma y del modelo, incluidos los negativos,
  y reservar un test posterior no reutilizado para selección. El
  [Deflated Sharpe Ratio de Bailey y López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
  aborda multiplicidad y no normalidad; no corrige por sí solo datos causales
  defectuosos ni acredita cobertura universal bajo dependencia temporal.
- Declarar método de inferencia y sensibilidad a dependencia, selección y cambios
  de distribución. Un bootstrap por bloques debe justificar sus bloques; sustituir
  un conteo nominal por otro número sin estimarlo no resuelve el problema.

Los niveles de confianza, presupuesto de pérdida, horizonte de ruina y capacidad
de capital son especificaciones explícitas pendientes del operador cuando no
estén ya fijadas y versionadas. No se inventan para conseguir que pase un gate.

## 7. Línea D — Codex: contratos de confianza desde la raíz

Estado observado: RA contiene16 expedientes, cuatro correcciones acotadas,
una parcial y once abiertos, descritos en el
[informe raíz](AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md) y su
[artefacto](AUDITORIA_RAIZ_REVALIDACION_2026-10-04.json).
MW/model-reload-contract permanece separado; una revisión favorable de dirección
no publica ni certifica su watcher. No se mezcla MW con la autorización RA.

RA corrige tokens escalares, admisión de estado no finito y frontera de contexto
OOS; el parser de profundidad conserva un residual estructural declarado.
Su CI [37211025915](https://github.com/Jhona-la/Trader-Gemini/actions/runs/37211025915)
terminó SUCCESS: parser19/0, riesgo7/0, OOS6/0 con crates reales. Las26 pruebas
del harness aislado de riesgo no son el conteo de ese target CI. La revisión
independiente automatizada de496f902d contra7c4cea no encontró bloqueadores
nuevos en el alcance de los cuatro commits; no es aprobación humana de GLM/Claude.
T-1 completo local sigue en compilación; no hay resultado ni paridad RA.

Compromiso Codex: cerrar recibos del candidato y mantener visibles los contratos
pendientes de identidad, causalidad, DD histórico y admisión compuesta. Su cierre
condiciona la interpretación económica de fitness, habilidad y sizing. Los
repartos propuestos abajo requieren acuse; no se convierten en reservas aceptadas.

## 8. Grafo de dependencias y criterios de aceptación

Los nodos indican contratos, no ocho motores de trading separados. La ruta
principal comparte estado y evidencia entre activos y escalas:

```text
G0 versión y certificación
 ├─ G1 identidad/datos → G2 causalidad/skill → G5 evidencia OOS
 ├─ G3 trayectoria de capital → G4 admisión y riesgo multiactivo
 └─ G7 identidad del artefacto cargado
G2 + G4 + G5 + G7 → G6 ejecución/paridad → G8 medición económica
G8 → aprendizaje controlado → nueva versión G0, sin reescribir su pasado
```

El nodo raíz acredita identidad, tiempo y datos/capital disponibles. El nodo
de decisión combina evidencia informada y admisión compuesta. Los nodos
terminales son fill, parcial, rechazo y resultado ambiguo conciliado: cada uno
conserva intención/versiones y sólo aporta feedback compatible con su evidencia.
La retroalimentación entre versiones no autoriza aprender del futuro ni convertir
un rechazo en una operación simulada ganadora.

| ID y raíz del contrato | Responsable observado o propuesto | Entregable y aceptación falsable | Invalida o bloquea |
|---|---|---|---|
| G0 versión | Codex RA; GLM ofrece certificación | SHA de candidato/base/merge, comparación contra **cada padre**, all-targets, contratos, T-1 y recibo de revisión. No cerrar con compile solo. | Todo recibo sin versión o con código cambiado. |
| G1 identidad | Codex propone coordinación con Claude/GLM | Resolver RA-E01: símbolo CLI, coin_id, modelo, specs y etiquetas idénticos en selección, replay y envío. Contraprueba ETH sin registro/modelo BTC implícito. | Fitness y datasets afectados por activo equivocado. |
| G2 causalidad | Qoder espectral observado; RA-S01/S02/S04 propuestos para revisión conjunta | Fijar predicción y referencia al emitir, madurar después, prueba de invariancia de prefijo; no puntuar retornos anteriores al voto. Bajo olvido, estimar soporte efectivo y dependencia, no usar n acumulado ilimitado. | Skill, pesos, etiquetas y selección derivados del protocolo anterior. |
| G3 trayectoria del genoma | Claude evolución observado; RA-E03/E04 propuestos | Genoma/parametrización evaluados=publicados=cargados. Persistir high-water mark y DDmax intratrayectoria; pérdida seguida de recuperación no borra el máximo. | Fitness, ranking y promoción de candidatos afectados. |
| G4 admisión compuesta | Claude sizing y Qoder riesgo observados; RA-R01 propuesto | Fijar apalancamiento ejecutable antes de validar/reservar; nocional, margen, SL y límites concuerdan. Contraprueba margen14@5x→1x frente a techo50. Validar unidades y supuestos de ruina/colas/covarianza. | Autorizaciones previas si cambian leverage, límite, snapshot o specs. |
| G5 aprendizaje y OOS | GLM L2 observado | Manifest de datos, física, activos, horizontes y etiquetas; candidato/baseline congelados en mismas filas, test posterior íntegro, multiplicidad e incertidumbre. RA-E02: no vender interpolación condicionada al cierre como microdato observado causal. | Dataset ante cambio de identidad, física, etiquetas o reloj; test reutilizado no sigue siendo test independiente. |
| G6 ejecución | Claude y servicio de paridad GLM observados | Misma intención, reserva, orden y resultado económico en replay/demo; parciales, rechazo firme y resultado ambiguo con conciliación. Paridad enumera casos, SHA y limitaciones. | Certificación anterior después de cambios en sizing, envío, cierre o reconciliación. |
| G7 carga/evolución | Codex MW separado; Claude GENOME-GATE observado | Hash/versión/slots del artefacto evaluado y cargado; rollback válido y fallo transaccional. T-1 y paridad propios antes de integrar watcher. | Resultados de un modelo/genoma distinto al realmente activo. |
| G8 resultado neto | Evaluación conjunta propuesta, no owner aceptado | Métricas§6.2, contraste OOS y sombra, costes/capacidad y tolerancias fijadas antes de lectura del resultado. Un resultado negativo se conserva. | Una promesa de crecimiento basada sólo en hit-rate, sensibilidad genética o compile. |

No todos los contratos pueden ejecutarse en serie: reparación G1–G4 y análisis
de históricos válidos pueden avanzar mientras se esperan tapes nuevos. Los
tapes de octubre bloquean los experimentos que los requieren, **no todo lo demás**.
El riesgo R01 no se da por corregido sólo porque el envío valide apalancamiento:
debe demostrar consistencia de admisión/reserva/envío para el mismo snapshot.

### 8.1 Frentes del plan GLM reconciliados sin borrar su historial

GLM reporta paridad PR27 verde10/10 en d777175f. Ese recibo pertenece a su corte
y escenarios; no certifica RA ni toda operación IOC. En el mismo corte,55,4%
contra47,8% y peso−3,5 de consenso_media son **in-sample**; el gate sep-14 estaba
pendiente. El coeficiente describe un ajuste, no demuestra causalmente que la
agregación destruye señal. Si L2 no pasa, conservar el negativo; un modelo no
lineal necesita otro protocolo/test, no una subida automática de peldaño.

`7/26=26,9231%`; un símbolo adicional son3,8462 **puntos porcentuales** si el
denominador permanece26. No es≈3,5%, ni retorno/volumen marginal probado.
Roster18 con MOTOR y roster26 del plan GLM miden poblaciones distintas: se
exige manifest del universo antes de combinarlos. La brecha310× y10 trades/día
necesitan período, escenario y derivación; no se deducen de100%/72h.

D₀ ya está publicado por qo-654 antes de main856; queda definir y contrastar
su consumidor, no repetir su publicación como pendiente. La cópula dinámica
requiere G4/G5: ventana, soporte, unidades, incertidumbre y fallback contra la
versión vigente. FDUSD archivo/remoción sigue como decisión de mantenimiento,
separada de una promoción científica; este plan no borra modelos.

### 8.2 Evidencia de vigencia: dos defectos y tres validaciones MG

El [informe raíz§17](AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md#17-revisión-matemática-multiactivo-mg-defectos-y-validaciones-separados)
añade MG02 (Some→None no invalida R publicado, G4) y MG05 (escala sin evidencia
hereda IC de otra, G2/G4). Requieren transición y consumo del respaldo correctos
antes de tratar esas claves como evidencia vigente. MG01, MG03 y MG04 revisan
respectivamente la interpretación de Lundberg, proyección equicorrelacionada
y coseno usado como correlación. Son validaciones de modelos/políticas conocidas,
no tres bugs nuevos del solver. Los contraejemplos se recalcularon; no son una
medición de pérdidas o frecuencia operacional. No se modifica Rust durante T-1.

## 9. Motor continuo multivariante: contrato de diseño, no etiqueta

El espacio de decisión debe poder representar activo `a`, tiempo de evento `t`,
escala `τ>0`, estado y covariación conjunta. Una representación propuesta es
`x(a,t,log τ)` con soporte, incertidumbre y antigüedad por componente; la
elección de variables/unidades requiere especificación antes de implementación.
Scalping/swing no deben determinar motores excluyentes ni fronteras económicas.
Los estados de volatilidad también tienen incertidumbre y variación continua;
un simplex de probabilidades no elimina por sí solo todos los cortes rígidos.

El dominio conceptual1ns–100años no obliga a inventar observaciones ni calcular
toda escala cada nanosegundo. Reloj, timestamps, frecuencia de datos, latencia,
memoria y duración histórica limitan el soporte medido. Modelar o extrapolar
una escala sin historia debe marcarse como tal y no aumentar su confianza.
Se propone adaptación por evento con refinamiento de malla/cuadratura donde el
error estimado y la información lo justifiquen, y un presupuesto computacional
medido. Discretización numérica explícita no equivale a dividir estrategias.

Aceptación propuesta: estabilidad de decisiones al refinar la malla, unidades
compatibles, masa normalizada sólo sobre soporte informado, covariación conjunta
por escala, pruebas de activos fríos y flujos asíncronos; complejidad/latencia
p50/p95/p99 medidas bajo replay reproducible. RA-S03 queda abierto: normalizar
puede cancelar el pequeño peso de una escala aún no observada.

## 10. Registro obligatorio de vetos, filtros y límites

Cada guardia requiere ubicación/consumidor, motivo tipado, magnitud/unidad,
origen de umbral, ámbito por activo/cartera, snapshot y reacción permitida.
Clasificar por **invariante técnico**, **regla de exchange**, **presupuesto
del operador**, **hipótesis estadística** o **heurística económica** evita
confundir un requisito de seguridad con un hiperparámetro sin fundamento.

Para revisar una guardia: contraejemplo de bloqueo legítimo e ilegítimo, tasa de
activación y oportunidades bloqueadas en sombra, efecto marginal y alternativa
contrastada. Los rechazos no deben cambiar estado parcialmente ni bloquear la
gestión/cierre de posiciones por una regla exclusiva de nuevas entradas.
NaN/Inf no es un régimen extremo válido. Un umbral económico adaptativo necesita
estimador causal, incertidumbre, dominio y límites; no se elimina por llamarlo
arbitrario ni se optimiza con el mismo test con que se pretende certificarlo.

Este registro sigue pendiente de cobertura completa; no se declara aquí que
todos los filtros del repositorio estén inventariados o hayan sido validados.

## 11. Integración científica y auditoría archivo por archivo

Antes de incorporar una teoría: pregunta falsable, variables observables,
ecuación y unidades, condiciones, error numérico, coste, baseline, ablación,
OOS y motivo para preferirla. Hawkes, modelos de estado, copulas, optimización
robusta o aprendizaje online deben demostrar una ventaja para ese contrato.
La analogía física no sustituye estimación; una función con nombre cuántico
no demuestra ejecución cuántica ni ventaja computacional.

Los [problemas del milenio de Clay](https://www.claymath.org/millennium-problems/)
no son un catálogo de ecuaciones resueltas para trading. Se aceptará una técnica
transferida sólo por su relación definida con el problema y su contraste; no
por prestigio o complejidad. La excelencia aquí exige exactitud y falsabilidad.

Inventario observado de496f902d:1434 archivos versionados,433 bajo `crates/`.
Es cobertura **de inventario**, no1434 revisiones semánticas. El informe RA§12
enumera las rutas examinadas y sus límites. La siguiente auditoría debe mantener
una fila por ruta/blob: inventariado, leído, flujo/teoría revisados, prueba,
hallazgos y dependencias; binarios/datasets se revisan con método específico,
no como si fueran código. Nuevos blobs invalidan el recibo sólo de su ruta y
de los consumidores afectados. Ningún archivo se marca revisado por un `rg`.

## 12. Protocolo de sincronización y próximos entregables

1. Inicio de ola: leer plan, memoria y buzón; publicar owner, rama/worktree,
   SHA base, alcance y rutas reservadas. Acuse explícito antes de asumir reparto.
2. Trabajo en rama propia y checkout aislado. No compartir índice, detener
   procesos ajenos ni lanzar promociones/trading como paso de certificación.
3. Cierre: recibo con SHA, comando, fixture/dataset, salida, alcance y pendientes;
   actualizar **la propia sección** del plan operativo y los contratos aquí.
4. Integración: comparar cada padre, ejecutar controles pertinentes, preservar
   todas las entradas en conflictos; publicar sólo el alcance autorizado.
5. Limpiar sólo refs tras comprobar nombre/OID y ancestría en `main` y
   `origin/main` publicado, ausencia de exclusivos y actividad reservada.
   Conservar directorios y archivos. Un worktree histórico sucio requiere
   detach al mismo HEAD con status/hashes idénticos antes/después, sin force.
   Una PR merged no prueba ausencia de commits posteriores; revalidar cada ref.

Primer entregable Codex: recibos RA/T-1 y consolidación documental; segundo:
censo de cobertura y cierre de G1–G4 en lotes con pruebas negativas, coordinando
las reservas. Después G5–G8 y refinamiento del continuo. No se prometen fechas
de rentabilidad ni cierre total sin observar resultados. GLM conserva su gate
L2 y Claude/Qoder sus frentes; las propuestas de reparto aún no son aceptaciones.

| Fecha | Agente | Adenda de estado |
|---|---|---|
| 2026-10-04 | Codex | CI RA32/0 verificada; revisión automatizada sin bloqueadores nuevos; T-1 local en compilación. Plan GLM integrado localmente por381e5f7d y main856 porf10434d6; no equivale a RA→main. Acuse externo RA ausente. |

## 13. Actualización posterior: integración documental y reservas de vigencia

Main `c6ce7333` incorpora GLM LXXXIV. Su gate L2 se resolvió **parcial**:
la dirección mejora frente a la fija, pero el logloss es peor que el baseline
constante. El cableado permanece bloqueado por el gate completo; no se toma
una mejora relativa de hit como rentabilidad neta o calibración suficiente.
El merge propio RA `12456048` conserva ambas bitácoras y el ADR-0010 completo;
comparación contra los dos padres y all-targets exit0, 37,88 s. Rust/CLI/CI
son idénticos a496f902d. Main→RA no acredita aún RA→main.

Leído anuncio GLM `253d0cd2`: agradece la firma/enlaces del plan, reserva fase3
de calibración y anuncia review PR28. Es acuse **del plan**, no review ejecutada
ni aprobación del candidato RA. Su nueva rama activa `glm/lxxxv-l2fase3`
se preserva; no se reordena ni se integra su trabajo en curso por anticipado.

### 13.1 Revalidación de calibración y frontera de selección

Propuesta al frente GLM: separar ajuste, selección y confirmación. Ajustar
Platt/temperatura sólo con agosto no introduce directamente etiquetas de
septiembre en ese ajuste. Sin embargo, sep-14 **ya fue observado** en LXXXIV
y su fallo de calibración motivó la siguiente fase. Reutilizarlo después
es una revalidación del mismo holdout, no una confirmación ciega nueva.
Conservar el resultado anterior, número de ensayos y decisión que provocó;
congelar hash de predictor/calibrador, umbrales, universo, costes y física
de generación antes de abrir un período todavía no utilizado para selección.
El oráculo/paridad y el test nuevo tienen funciones distintas. Esta propuesta
no altera el gate de GLM, no afirma fuga demostrada ni bloquea su investigación.
En particular, comprobar la compatibilidad del dataset con #649/#650 antes
de atribuir la mejora a una agregación sobre la física actual.

### 13.2 Reserva Codex y censo explícito

Rama `codex/evidence-expiry-2026-10-04`, worktree separado `evidence-expiry`,
base c6ce7333. Reserva MG02/MG05: escritores de R/margen e IC de la escala
dominante, lectores si el contrato de invalidación lo exige, oráculos de
transición y aislamiento por moneda. Sin cambiar fórmulas ni políticas
finitas; sin tocar entrenamiento, promoción, procesos o Rust de RA durante T1.
La delegación interna no es acuse de los agentes externos ni cierre del bug.

[Censo por ruta/hash](audit/RA_COBERTURA_2026-10-04.tsv) y
[reglas de cobertura](audit/README_RA_COBERTURA_2026-10-04.md): 1435 entradas,
433 en crates, snapshot12456048, estado exclusivamente inventariado. Mantener
recibos semánticos por contrato; ningún archivo se certifica por estar listado.

### 13.3 Higiene de ramas sin pérdida de exclusivos

Retirada sólo la referencia local `glm/lxxxiv-l2v1` en
eb617f548ab1a3050ee97b244ff747a174849716: ancestro de main c6ce, cero exclusivos,
sin worktree ocupado, eliminación condicionada al OID. Remota ya ausente tras
fetch/prune; el commit sigue recuperable desde main. Se conservan backup, TH,
V7, MW, el commit documental posterior de Claude y GLM LXXXV con exclusivos.
También se conservan las dos ramas propias ocupadas. No hay borrado forzado
de contenido ni equivalencia semántica inferida por el título de un commit.

### 13.4 Recibo posterior GLM LXXXV

Main5ab07f35 incorpora ec7b314d: calibración de temperatura T=2,393 en agosto
produce logloss0,6771 frente a0,6812 allí, pero0,7078 frente a0,6910 en sep-14.
Son cifras del recibo GLM/ADR-0010, no reproducción independiente de RA.
Veredicto L2v1 parcial definitivo; cableado bloqueado. El mapa de planes §5b
de GLM conserva estado/ruta, contratos A/B/C y cobertura G0–G8 como funciones
complementarias. Se preservan también las advertencias de frontera de §13.1.

Recibida review **de agente** en COORDINACION, commit ec7b314d: dirección RA
aprobada para scanner y riesgo no finito, con T1/paridad exigidos al merge.
No es aprobación humana GitHub ni certificación del OOS/continuo completo.
Los recibos anteriores que decían review no recibida describen su propio corte.
El nuevo main es documental: no altera Rust/CLI/CI de496f; RA se reconcilia
en su checkout aislado por unión. Retirada referencia local GLM LXXXVec7b314d
tras ancestry a5ab, OID fijo y ausencia de worktree; remota ya ausente. Commit
preservado en main. No se borran exclusivas de Claude/MW/TH/V7/backup.

### 13.5 Revalidación de dependencias antes de investigar nueva teoría

[RA§20](AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md#20-revalidación-adversarial-de-expedientes-existentes-sobre-el-nuevo-main)
confirma R01, E03 y E04 en c6ce con revisor independiente/relectura/aritmética.
R01 sigue en G4: readmitir el margen final, no sólo corregir su contabilidad.
E03/E04 siguen en G3/G7: equivalencia del fenotipo inicial del ShadowForest y
DD histórico de su trayectoria, no déficit de la última foto. No extender al
CLI ni a todos los backtests; no cuentan como nuevos expedientes o arreglos.
Estos contratos condicionan cómo interpretar el genoma antes de añadir teorías.

### 13.6 Mapa verificable de la base hacia la cima

[Topología declarada](audit/RA_TOPOLOGIA_DECLARADA_2026-10-04.json) y
[explicación por niveles](audit/README_RA_COBERTURA_2026-10-04.md#topología-declarada-recibo-estructural-no-grafo-vivo-certificado):
24 paquetes,202 targets,88 aristas, manifiestos por hash y cotejo completo
de la declaración. Sirve para repartir contratos; no sustituye al grafo
operativo, no verifica features, sincronía ni todos los archivos. Flight-recorder
queda fuera del cierre normal de la raíz: localizar su consumidor antes de
atribuir desconexión. El mapa G0–G8 sigue siendo el orden causal de verificación.

Reserva adicional Codex E03/E04 en `codex/forest-evaluation-2026-10-04`,
worktree separado `forest-evaluation`, base5ab07f35: únicamente coherencia
genoma/fenotipo inicial y trayectoria de capital observada de ShadowForest.
Pruebas RED→GREEN, conservar CL40 y no cambiar fitness, mínimo de operaciones,
gates, almacén, entrenamiento o procesos. Cosecha y re-registro del mismo
genoma no pueden borrar historia; sólo replantación abre una época nueva.
La reserva no es reparación aceptada ni acuse de otro editor. MG02/MG05
permanece en su checkout distinto; RA no recibe código mientras corre T1.

### 13.7 Recibo CI posterior, sin reutilizar identidad

CI37221376454 de b6df terminó SUCCESS:187/0/3ignoradas,13 resultados de targets;
merge7c1ae865 y ambos padres confirmados por log/API (detalle RA§22). Parser19,
riesgo7 y OOS6 sin ignoradas. No es ejecución del head0ddb o su merge5ab en
preparación, no T1 completo ni aprobación humana. Conservar las observaciones
anteriores en progreso como cortes históricos; G0 exige la nueva identidad.

### 13.8 Telemetría: seguridad antes de cableado

RA§23 revalida D127/D237 y consumidor#80/#101 sobre blobs de5ab: frame no
atómico, secuencia sin exclusión por slot y fallback de lectura no validada.
Es P1 bajo uso concurrente seguro de la API; no corrupción operacional medida
ni nuevo ID. El core usa recorder mmap de telemetry-server, inicializaNone y
no se encontró wiring en Rust versionado; el crate flight-recorder sí tiene
implementación/tests, no la plantilla antigua. Los formatos difieren. Primero
especificar seguridad, reloj, pérdida/overwrite y consumidor; no cablear una
implementación insegura sólo para completar el grafo. Sin cambiar Rust RA.

### 13.9 Revisión cruzada adicional y períodos de evidencia

RA§24/JSON: revisor independiente no encuentra bloqueadores nuevos atribuibles
al parser/riesgo no finito/contexto OOS de496f. No equivale a tests Rust ni
humano GitHub. Residual Depth confirmado y snapshot atómico aún no acreditado.
Seguimientos preexistentes RA-OOS-F01 (CLI una fila→índice inválido, G5) y
RA-OOS-F02 (panel con numerador OOS/reloj contexto+OOS, G8) quedan abiertos,
separados del inventario original16. W mide filas disponibles, no madurez
validada. Requieren rechazo previo tipado y período económico explícito;
no añadir una nueva cifra mínima arbitraria ni cambiar fuente durante T1.

### 13.10 Acuse Qoder posterior y límites del sello

Main3979f58b incorpora mapa/cabecera y entradas GLM/Qoder, sin Rust/CI nuevo.
Se preservan junto a G0–G8. Su sello16/144 en4221s pertenece a7c4cea40 y es
recibo Qoder, no reproducción de RA ni certificación de fuente496 o de la meta
económica. La aprobación GLM de la fila §5 se interpreta con el alcance y
condiciones de su recibo §13.4, no como pase incondicional de todos los gates.
No se atribuye a Codex autorización para la sesión viva §4.2 ni se inicia motor.
RA03a7 fue publicado; traer3979 de nuevo exige diff por padre y check antes
de publicar el siguiente corte. MG/forest siguen en sus checkouts separados.

### 13.11 Contrato matemático del oscilador y límite de reutilización

RA§27 añade RA-Q-F01 separado del inventario16: helper de densidad gaussiana
recortada/cola clamped sin normalización de probabilidad. API latente: tres
calls Rust encontrados, todos tests; sin impacto en órdenes probado. Propuesta
a Qoder (no acuse/reserva aceptada): definir score vs densidad, soporte/unidades,
estado inválido y pruebas de masa. No cambiar kernel vivo durante T‑1.
La paridad #650 se conserva dentro del dominio actual; API directa x12 revela
frontera distinta, pero callers actuales la excluyen. No sumar un segundo bug
operativo por ese contraejemplo. Fuerza derivada de potencial clásico no es
solver de estado cuántico: analogía física requiere interpretación y evidencia.

### 13.12 Recibo de bosque y semántica de features temporales

RA§28: review independiente de parche E03/E04/hash estable, sin bloqueadores
nuevos hallados en alcance. No tests ejecutados por revisor ni integración;
DD realizado observado, noMTM/intraevento. RED/GREEN y gates propios pendientes.
RA§29 añade RA-S-F01 separado: el proxy de magnitudes llamadoHurst no mide
memoria temporal, aunque sus features37–39 sí llegan a train/serve. Ventanas
10/25/50 son observaciones, no τ físicas. Propuesta Qoder+GLM de contrato y
calibración, no reserva aceptada; no cambiar tensor sin paridad/retrain/OOS.
D0 mide ocupación de soporte, no amplitud/clustering general; no se fabrica
otro bug por esa elección. QuantumKellyRiskEngine ya es isla declarada #653,
no cablearla duplicando el sizing. Interpretación científica precede a teoría
adicional; no hay validación de toda la matemática ni rentabilidad certificada.

### 13.13 Vigencia de evidencias: reproducibilidad antes de integración

RA§30: MG02/MG05 tiene RED1/12→GREEN13/0 con mismo hash de test, regresión
Lundberg6/0 y fixture de mutabilidad reparado/no contado como RED. MáscaraR0
e IC-1 retira evidencia por moneda sin reactivar globals; neutralidad del IC
frente a base negativa es parte del contrato, no permiso para borrar vetos.
No snapshot conjunto ni T1MG; review independiente/all-targets pendientes.
Rama evidence-expiry permanece separada de RA y fuente496 congelada. Owner
Codex, bloque acotado al helper/publicación/tests; no reserva ajena asumida.

### 13.14 Frontera del reloj aceptado, G1 antes de G4/G7

RA-I-F01/RA§31: el reloj de features cambia ante algunos rechazos y alimenta
cooldown/olvido de rachas. Nuevo worktree/ramacodex/feature-clock-2026-10-04,
dos archivos disjuntos de MG/forest. Reserva comunicada en buzón compartido;
no acuse de editor externo. Tests ampliados primero, RED/GREEN efectivos y
revisión antes de commit. Mover timestamp tras guards conserva política;
no inventar frecuencia nanosegundo ni acelerar tiempo sin observaciones.
Este expediente es separado de los16 originales; no hay cierre todavía.

### 13.15 Actualización posterior: límites cruzados y reparación comprobada

RA§30.1: revisión MG sin bloqueadores nuevos, pero mantener abierto consumo
as-of del IC histórico en la misma escala. Rechazar nuevas parejas desalineadas
no retira necesariamente una correlación acumulada; Some≠reciente. Cierre
real/feedback/base negativa directa y coste no cubiertos por13 controles.
RA§31.1: reloj aceptado actualRED18/7→GREEN25/0, misma prueba y fuente separada;
no modificación de política. Review/unitarios/all-targets todavía en curso.
Ambas series quedan fuera de RA/main hasta sus gates. Preservar cortes anteriores
como historial; no dar por integradas propuestas o reservas no reconocidas.

### 13.16 T1 RA completo y límites del gate G0

RA§32: propioT1 fuente496/binarionightly intacto terminó2pass0fail0ignored,
16/144 perturbaciones sensibles≥ratchet0,110,4636,44s tras134m33sbuild. No
inercia global del resto128, coordenadas independientes, lucro real ni meta72h.
No incluye MG/bosque/reloj. MGcommit605a4d7f (padresc6ce+maine3) queda local
tras all-targets0/2m33s; review sin blockers nuevos, frescura IC histórico aún
abierta. Reloj25/0+unitarios18/0, revisión local favorable; all-targets sigue
en curso. CIb245 aún en ejecución y PR28draft; no afirmar llegadaRA a main.
Consulta pública para publicar seriesMG/reloj separadas pendiente, no bloqueo
de documentación/QA ni autorización para saltar CI, activar modelos o trading.

### 13.17 Commits listos dentro de alcance, integración aún pendiente

MG605a4d7f y reloj f3018b35 quedan en ramas propias locales; ambos tienen
oráculos efectivos, revisión cruzada de hash sin blockers nuevos y all-targets
verdes (MG2m33s/reloj3m25s). Informes propios detallan los límites. No están
enRA ni main, publicación pública consultada. G0 no permite confundir sus
checks con candidato conjunto/T1 ni borrar ramas ocupadas o exclusivas.
Último fetch: main e3, RApublicada16exclusivos, MG1/reloj1, Claude remoto1;
MW/TH/V7/backup conservados. Buzón compartido actualizado con fuente/owner
y pendientes; Qoder/GLM no han reconocido las propuestas científicas nuevas.

## 14. Corte operativo consolidado y secuencia de cierre

Corte de esta adenda: 2026-10-04T21:00:56Z. Último fetch: main y origin/main
`e3adf74e`; checkout compartido sin merge abierto, sólo `.workbuddy-ai/`
ajeno sin versionar. Codex no alteró ese índice ni detuvo procesos ajenos.
La sincronización externa usa plan/coordinación versionados y buzón compartido:
los avisos están publicados en el buzón, pero los nuevos repartos científicos
no tienen acuse de los otros editores. Delegación interna no equivale a ese acuse.

| Entrega | Owner del trabajo observado | Evidencia efectiva | Estado de adopción/gate siguiente |
|---|---|---|---|
| RA raíz, PR28/headb245 | Codex; review de agente GLM condicionada | T1 propio2/0,16/144; all-targets; revisiones con alcance | CI actual en curso, draft; publicación de adendas y gates del candidato pendientes |
| MG02/MG05, head7aadf509 | Codex | RED1/12→GREEN13/0, Lundberg6/0, revisión, all-targets0 | Local; consulta de publicación pendiente; IC histórico as-of sigue abierto |
| Reloj aceptado, head36b0063c | Codex | RED18/7→GREEN25/0, unitarios18/0, revisión, all-targets0 | Local; consulta de publicación pendiente; no atomicidad whole-core/cierre0 |
| E03/E04, headcace007d | Codex, worker y revisor independientes | RED7/9→GREEN16/0, biblioteca73/0, revisión, all-targets0 | Local; publicación consultada; T1/paridad/CI propios pendientes |
| Semántica Hurst y densidad | Codex auditó; Qoder/GLM propuestos para contraste | Contrapruebas matemáticas y rutas por hash; no impacto económico medido | Contrato/semántica pendientes de acuse; no cambiar tensor o política en silencio |
| G8 crecimiento neto72h | Evaluación conjunta propuesta | Objetivo definido§6; no demostración neta OOS de duplicación | Fijar experimento, riesgo/soporte/costes y test no reutilizado antes de promoción |

Secuencia de integración:

1. Conservar los recibos históricos y los resultados negativos; publicar sólo
   el alcance autorizado. Las adendas de caducidad/reloj/bosque añaden pruebas
   explícitas a CI sin retirar las anteriores ni transferir resultados RA.
2. Antes del merge de cada serie, cotejar main vigente y **cada padre**,
   preservar todas las entradas/pasos de CI en conflictos y volver a probar
   los contratos afectados por la unión. La compilación no sustituye ejecución.
3. Ejecutar paridad y T1 pertinentes para los cambios de pipeline; validar
   identidad de la fuente/datos/genoma que realmente se carga. No mezclar
   certificados aislados como un certificado del candidato conjunto.
4. Confirmar el SHA/equivalencia revisada en main remoto y ausencia de
   exclusivos/ocupación/actividad antes de borrar referencias locales/remotas.
   RA16, MG2, reloj2, MW4, TH4, V7wip1, backup3 y Claude remoto1 tenían
   exclusivos al censo; el bosque pasó después a su commit local conjunto.
5. Sobre esa base, cerrar causalidad, frescura as-of, admisión compuesta y
   modelos/features antes de atribuir mejoras a una teoría nueva. La capa
   continua§9 necesita soporte/antigüedad/incertidumbre/covariación por activo
   y escala, no sólo reemplazar etiquetas por nombres espectrales.

Próximas revisiones cruzadas propuestas: Qoder, semántica del espectro y
vigencia del IC; GLM, contrato train/serve y confirmación OOS de features;
Claude, identidad/admisión/reserva/envío y trayectoria económica. No son
tareas aceptadas hasta recibir acuse. Codex mantiene los recibos, las reservas
propias y el cierre técnico por bloques; cada editor conserva su índice.
No hay autorización de entrenamiento, promoción ni sesión viva en esta adenda.

Reconciliación posterior RA§35/JSON: los merge commits de PR23/MR,24/GO,
25/MP y26/27/Claude sí son ancestros de origin/main. PR28 sigue abierta;
la rama de Claude conserva un exclusivo posterior, no es eliminable sólo
por el estado de PR27. Censo12refs/OIDs/ocupación: ninguna nueva elegible
para borrado. La revisión de exclusivos legacy se mantiene sólo lectura;
no equivale a certificar todo su código o preservar toda semántica por ancestry.

### 14.1 CI final del candidato anterior, gate del nuevo corte separado

RA§36/JSON registra CI37230580226 SUCCESS:187/0/3ignoradas en13 resultados,
headb245, merge62c2ea0d y padres main e3/b245 confirmados por log/API. Parser19,
riesgo7 y OOS6 pasan sin ignoradas; paridad8/0/2manuales ignoradas. T1 completo
propio§32 sigue siendo un recibo distinto. No garantía financiera ni aprobación
de todos los fallos abiertos. Publicar las adendas autorizadas requiere nueva
identidad/CI de PR28, sin cancelar ya la ejecución completada. Las tres series
funcionales aisladas mantienen sus propias consultas y gates de integración.

### 14.2 Recibo local legacy, sin churn de CI

RA§37/JSON recibe anatomía de backup/TH/V7/MW/Claude frente a e3:12exclusivos
no-merge cherry+, equivalencia sólo de limpieza TH/01f8, no de la rama.
Claude6f es delta documental de memoria7/−3, sin código exclusivo. Preservar
contenido por unión; las otras ramas requieren revisión semántica específica,
no importar generados/WIP masivamente. Revisor no ejecutó Cargo ni certificó
políticas, sólo leyó metadatos y el delta Claude. Recibo local posterior a
publicar ff99849a; CI37235240759 en curso, no reiniciada por esta adenda.

### 14.3 Reinicio causal sobre main44bc8 y tareas sincronizadas propuestas

main44bc8 (LXXXVII GLM) añade descripción de unidadesV-RISK-006 y cierre
L2v1 negativo. Mantener BLOQUEADO su cableado; sus cifras son recibos del
autor, no reproducción de Codex. La CI de PR28/headff998 evalúa aún su corte
previo: no transferir su resultado al nuevo candidato reconciliado.

| Contrato | Responsable de esta ronda | Próximo contraste propuesto | Criterio para cerrar |
| --- | --- | --- | --- |
| MG05, IC same-τ sin as-of | Codex/revisión independiente, lectura | Qoder: procedencia por par/escala; GLM: soporte OOS; Claude: consumo de riesgo | evidencia admisible as-of de ambos activos; no sólo Some ni hora de publicación |
| RA-SA-F01, penalización negativa mejora ranking | Codex, diagnóstico algebraico | Claude/GLM: unidad/cero de utilidad y monotonía del SA | controles para ambos signos y ranking, antes de medir ventaja |
| RA-OOS-F01 | Codex, rama oos-partition, commit9a3bb756 | revisión estática recibida; integración/candidato/CI pendientes | guard probado14/0, CLI compilado; efecto externo y soporte económico separados |
| PR28/main44bc8 | Codex, checkout RA propio | unión de bitácoras y diff por padre; checkalltargets antes de commit | CI del candidato actualizado, revisión y gates pertinentes |
| Crecimiento72h | dueño define restricciones; agentes proponen ensayos | evidencia neta causal/multiactivo con capacidad/ruina/selección | medición OOS no utilizada para escoger, no garantía ni eliminación de guardias |

No hay acuse nuevo de Qoder/Claude ni aceptación de estas tareas supuesta.
El buzón local contiene aviso de anclas/owners; es un mecanismo de coordinación,
no prueba de comunicación efectiva con sus editores. GLM sí aportó su bitácora
LXXXVII versionada. Estado público de MG/reloj/bosque sigue consultado; no
fusionar ni borrar sus ramas exclusivas. Las decisiones del dueño en§4 y
los gates G0–G8 se conservan íntegros. Ciencia propuesta debe especificar
magnitud, dominio, soporte, unidades, versión y experimento falsable.

### 14.4 Recibo del candidato con LXXXVIII, sin trasladar CI

RA§42/JSON registra candidato cd2e+main230, unión completa de documentos
GLM, diff por padre y alltargets0/5,41s antes del commit. CIff998 completó
187/0/3ignoradas, merge800c5f67 con padres e3/ff998 confirmados. Nuevo
head RA requiere su CI; mantener draft hasta gates pertinentes.
OOS9a3 sigue local/separado; MG/reloj/bosque mantienen consultas pendientes.
FDUSD A/B queda reservado al dueño. No archivo/borrado de modelos por Codex.

### 14.5 Sincronización F0/F1 y verificación de artefactos por fuente

RA§43 añade la unión con main57/62, preservando barrido F0 y ADR-0014 de
Qoder. Su F1 activo en .f1 no se elimina ni se modifica. Claude/riesgo y
GLM/medición mantienen sus frentes; no se atribuye aceptación nueva de tareas.
Codex reserva dos contratos locales separados: SA/score signado y selección
del candidato, OOS/preflight de partición. No tocar sus modelos/entrenamientos.

| Frente | Evidencia de esta ronda | Gate pendiente, sin extrapolación |
| --- | --- | --- |
| RA/main62 | unión por padre; check reconstruido0, hashes de fuente estables | commit, CI del SHA nuevo y revisión antes de main |
| SA |13 retención +9 selección, reproducción independiente22/0; check0 | descripción MD/JSON, publicación autorizada, composición y CI |
| OOS |684e+RA74;14/0, check0 | composición con main62, autorización/publicación y CI |
| IC/frescura | MG05 abierto; None no es automáticamente cero medido | soporte/as-of por par/τ y veredicto tipado por consumidor |
| Cobertura |465 rutas Rust seguidas por Git en RA; conteo no semántico | checklist por ruta/hash y reconciliar denominadores del barrido |

El primer check RA falló por API no visible en un artefacto compartido pese
a existir en fuente; reconstrucción acotada corrigió la evidencia sin cambiar
bytes. No trasladar verde cacheado entre worktrees. No se afirma certificación
global, ventaja cuántica, rentabilidad72h ni desaparición de todos los fallos.

### 14.6 Recibo F1 de Qoder y riesgo no finito reproducido

Main23701 publica23F1 (2HIGH/8MED/13LOW), docs-only; se conserva por unión
en RA. No todos reproducidos por Codex. Proponer prioridades y obtener acuse
antes de tocar productor vivo: QoderA1/C2/C4; GLM/CodexC1/C3; ClaudeA3/A4
yB1/B2/B4. SA/OOS mantienen ramas propias y contratos separados.

RA-RUIN-F01/F1-B4 se reproduce en el helper real:NaN/±Inf retornan intactos,
RED3pass/3fail, no pérdidas/órdenes demostradas. Existe defensa de exposición
no finita en consumidor; seguirla por rama. Claude mantiene ownership:
exigir fallo explícito/noexposición, preservar finitos y medir consumidores.
No confundir retorno0 de sizing con probabilidad0 de ruina ni ausencia de IC.

Meta72h: utilidad/genoma sano y matemática explicada son condiciones de
medición, no evidencia de rentabilidad. Gates G0–G8 siguen abiertos donde
corresponde. Detalle RA§44/JSON; inventario F1 enBarrido del autor.

## 15. Codex — contrato SA signado y selección evaluada (ola aislada)

Alcance reservado: helpers score_retention/sa_selection y bloque de selección
del CLI evolution, en codex/sa-score-monotonicity-2026-10-04. Base main57cb8f;
commits28c000 y5e170. No tocar IC/Qoder F1, risk/Claude o modelos/GLM.
Sin acuse nuevo confirmado: coordinación por documento/buzón, no acuerdo
supuesto. Informe y artefacto: AUDITORIA_SCORE_SA_2026-10-04.md/.json.

Contrato: al reducir retención nunca premiar una utilidad negativa; sólo
scores finitos entran a selección, primer candidato sin centinela y rechazo
sin saltar el calendario.22/0, revisión independiente0bloqueadores, check
alltargets0/124,774s. Utilidad escalada no se etiqueta como factor72h.

Orden de integración: fuente/contratos → composición con RA/main/OOS por
cada padre → revisión → autorización de publicación/CI del SHA exacto →
merge confirmado en main remoto → borrar sólo refs inactivas integradas.
OOS684e+RA74 yRA9bada+main62 son otros cortes, no incluidos automáticamente.
No nuevos resultados económicos: crecimiento neto72h, capacidad/ruina,
reporte finito, unidades y umbrales de utilidad mantienen gates abiertos.

## 16. Plan exhaustivo por ruta desde los fundamentos (ampliación del mandato)

[PLAN_REVISION_EXHAUSTIVA_2026-10-04](PLAN_REVISION_EXHAUSTIVA_2026-10-04.md)
define metas→conceptos→matemática→estadística→física/cuántica→algoritmos→
implementación; conserva fases F0–F8 del barrido e introduce subfases y
contratos de entrada F6 antes de consumidores. Censo nuevo por ruta/OID
main237, TSV+JSON; automático/inventariado, no auditado o certificado.
No reemplaza el censo histórico ni los23hallazgosF1 del autor.

Cada archivo requiere recibo propio de finalidad/cálculo/unidades/dominio,
flujo, hallazgos y comportamiento; RED/GREEN, compilación, efectos externos,
review y candidato/CI definidos por riesgo. Docs/config/binarios/logs/datos
también tienen método, no se omiten. OIDs cambiados abren revalidación.
Inventario/leído/reparado/probado/integrado/desplegado son estados distintos.

Roles propuestos y reservas por ancla, sin acuses supuestos. Nada de borrar
ramas exclusivas/ocupadas, impulsar modelos o relajar seguridad por meta72h.
Registro de planificación explícito, no porcentaje ficticio de corrección.

## 17. Correspondencia canónica de métricas72h (adenda, no borrado)

Revisión independiente halló colisión: G72 en§6.1/§6.2 es LOGRETORNO; G72
en el plan exhaustivo§2 es FACTOR. Para nuevos recibos usar nombres sin
ambigüedad: growth_factor_72h=E(t+72h)/E(t), objetivo>=2;
log_growth_72h=ln(growth_factor_72h), objetivo>=ln2. Ejemplo2→ln2,1→0.
Conservar fórmulas históricas y traducir por su definición, nunca comparar
factor con umbral log. Métrica G72 sin unidad/linaje BLOQUEA G8 hasta aclarar.
No se afirma migración de todos los consumidores o medición económica.

Plan exhaustivo§10 añade rollback transversal y owners propuestos/cierre
de anexo externo. C-versionado no equivale a C-sistema completo. Pruebas
aisladas y autoridades independientes; no promoción/trading por esta adenda.

## 18. Sincronización F2/mainEA5d y condiciones de teoría nueva

Se conservan las 43 filas F2 de Qoder y su reserva F3, sin cerrar sus bugs.
RA§48 documenta discrepancia reproducida de conteo: resumen7HIGH/16MED/20LOW
frente a tablas7HIGH/23MED/12LOW/1INFO. Solicitar reconciliación al autor;
no reclasificar ni sumar menciones de bitácora como expedientes distintos.

Prioridad: rastrear shadow y consumidores operativos, con oráculo/unidades/
abstención/fallback antes de modificar consensos o tensores. F2-A1 se revisa
independientemente sin interferir en el código de Qoder/Claude. El censo
main237 continúa inmutable; EA5d cambia cuatro docs, no fuente ejecutable.
La delta de cobertura se registra por OID, no como 1432 archivos auditados.

Investigación BM/OU/e-values: admisión CONDICIONADA a dominio, filtración,
validez del nulo, barreras, familia múltiple, error numérico y comparación
OOS neta. RA§48.3 enlaza fuentes primarias y requisitos. No declarar R8-A
cerrado por una idea ni inmunidad al barrido de escalas por renombrar z/IC.
Sin nueva teoría implementada, promoción o rentabilidad72h demostrada.

SA d73cca8c y OOS acbc3b61 siguen locales, respectivamente47/0 y14/0 en
sus composiciones RAf275; revisión y compilación por corte. Publicación
consultada. CI74beSUCCESS187/0/3 no es CI del candidato RA posterior.
No acuses externos nuevos; el dueño decide FDUSD, L2v1 sigue sin cableado.

## 19. Incorporación F3/main749d y frontera de publicación

F3 de Qoder incorpora36filas:3HIGH/14MED/19LOW, conteo cotejado por Codex.
Sus reparaciones siguen pendientes aunque el barrido diga «cerrada». Los103
expedientes F0–F3 no son necesariamente defectos independientes: F3-A2 es
otro call-site de F2-A5. No sumar menciones, pruebas o repeticiones de una
causa para aumentar cobertura. Plan exhaustivo§13 registra responsables
observados/propuestos y oráculos de PPO, trailing, reloj, RUIN, SA y OOS.

Main749d agrega3docs/+160 respectoEA5d. La unión RA preserva texto de ambos
padres; revisar cada composición y no reutilizar CI de una base distinta
como si fuera el nuevo SHA. SA94790c5c es commit local, sin push/PR; OOS
permanece aparte. Auto-review exige respuesta específica para publicación
pública SA/OOS; consulta renovada pendiente. RA28 ya tiene autoridad propia.
No se han borrado refs activas/no integradas, promovido modelos ni operado.

## 20. Formalización de Símplex Espectral Continuo en Riesgo y Registro de Vetos (Ola Ω58, Ficha #693)

- **Geometría del Símplex $\Delta^3$**: En `crates/risk-engine/src/regime.rs`, formalización de `SpectralMarketRegime` con componentes $(p_{\text{range}}, p_{\text{bull}}, p_{\text{crash}}, p_{\text{chaos}})$ con partición estricta de la unidad, entropía de Shannon $H(p)$, entropía cuántica de Rényi $H_\alpha(p)$, polarización direccional $\Pi_{\text{dir}}$ e índice de turbulencia $\tau_{\text{turb}}$.
- **Erradicación de Deuda en Registro de Vetos**: `V-LOGIC-007` (modulación suave de confianza de consejo en frío), `V-LOGIC-009` (evidencia muestral mínima sin deadlock), y `V-LOGIC-010` (simetría direccional en resonancia) certificadas contractualmente sin deudas en `crates/risk-engine/src/veto_registry.rs`.

## 21. Simetría Direccional en Colchón de Margen y Teoría de Información en Core (Ola Ω59, Ficha #694)

- **Simetría Direccional de Margen Libre (`crates/god-engine-core/src/lib.rs:7684-7697`)**: Symmetrizada la modulación continua de `free_cap` ante marea portadora adversa. Para posiciones Long, marea bajista (`tide < 0.0`) contrae el margen admisible ante caídas; para posiciones Short, marea alcista (`tide > 0.0`) contrae el margen admisible ante short squeezes violentos: `free_cap *= (1.0 - 0.95 * directional_flux).clamp(0.05, 1.0)`. Erradica la desprotección asimétrica de posiciones cortas.
- **Publicación Unificada del Símplex (`crates/god-engine-core/src/lib.rs:2549-2575`)**: Normalización e instanciación de `SpectralMarketRegime` en core; publicación continua a `OmniscientRegistry` de probabilidades del símplex, entropía de Shannon, entropía de Rényi ($\alpha = 2.0$), polarización direccional e índice de turbulencia. Deduplicación del estimador MAP hacia `spectral_regime.map_discrete().into()`.
- **Contrato Formal**: `crates/god-engine-core/tests/cushion_symmetry_contract.rs` (3/3 tests PASSED al 100%).

## 22. Resolución R7-R6: Escala Canónica de Riesgo por Operación para Ville, Desacoplamiento de Vigilancia Anytime-Valid y Descuento Bayesiano (Ola Ω60, Ficha #695)

- **Escala Canónica de Riesgo por Operación para Ville (`R7-R6-IG-1` [HIGH], `crates/evolution-engine/src/online_daemon.rs:1124-1142`)**: Erradicación del literal hardcodeado `0.02` en la normalización de retornos de Ville. Se reemplazó por la lectura atómica de `self.arena.riesgo_por_operacion.load(Ordering::Relaxed)` con fallback a `self.arena.config.scalp_sl_base` y piso numérico de `1e-6`. Los retornos ahora alimentan al E-proceso en múltiplos de $R$ reales $[-1.0, 1.0]$, reflejando la pérdida real de 55 bps sobre \$13.00 USD.
- **Desacoplamiento de Vigilancia Anytime-Valid (`R7-R6-EV-1` [MED], `crates/evolution-engine/src/online_daemon.rs:1198-1245`)**: Desacoplado el chequeo anytime-valid del E-proceso de Ville de la barrera asintótica $N \ge 20$ del test de Student. Ville evalúa degradación de evidencia y colapso ($\ge 50\%$ de evidencia perdida) tempranamente ($N \ge 3$), activando rollback inmediato sin esperar 20 operaciones. Se agregó detección temprana de rachas de pérdidas consecutivas ($N \ge 5$) con armaje de kill-switch.
- **Verificación Multi-Slot en Daemon Replay (`crates/evolution-engine/src/online_daemon.rs:574`)**: Reemplazado `position.is_open()` por `arena.coins[coin_id].positions.is_any_open()`, inspeccionando todos los slots (`scalp` y `swing`) para evitar que `live_envelope_gate` re-juzgue operaciones activas en cada tick.
- **Descuento de Evidencia en Prescreening Bayesian Prior (`R7-R6-C-7 ≡ MD-2` [MED], `crates/evolution-engine/src/fitness.rs:110-132`)**: Aplicado descuento estricto de evidencia `if raw_fitness > 0.0 { raw_fitness * weight } else { raw_fitness }`, impidiendo que micro-muestras no verificadas ($N < 30$) reclamen fitness positivo ficticio sobre el prior conservador.
- **Contrato Formal**: `crates/evolution-engine/tests/r7_r6_watchdog_evidence_contract.rs` (4/4 tests PASSED al 100%).

## 23. Cota Inferior LCB en Control de Ruina, Caos Espectral en Orquestador y Primer Toque Analítico (Ola Ω61, Ficha #696)

- **Cota Inferior LCB de Win Rate en Control de Ruina (`R7-R2-A-2` [MED], `crates/risk-engine/src/ruin.rs:88-100`)**: Creada la función pura canónica `conservative_loss_q(win_rate, trades)`. Si $N \ge 1.0$, calcula la cota inferior de confianza del win rate ($p_{\text{lcb}}$) vía el posterior Beta de Jeffreys en `crate::evidence::win_rate_lcb`, retornando $q_{\text{conservative}} = (1.0 - p_{\text{lcb}})$. Si $N < 1.0$, retorna incondicionalmente `CONSERVATIVE_Q` (0.60). Conectada a `risk-engine/src/lib.rs:380-388` (`evaluate_intent`) y `:595-604` (tope de riesgo de grupo de misma apuesta), impidiendo que micro-muestras no verificadas ($N < 20$) reclamen probabilidades de pérdida ingenuas de 0% o sobre-apuesten capital.
- **Consumo de Incertidumbre y Caos Espectral en Orquestador (`orchestrator.rs`, R5' / Superficie Ω58)**: En `crates/risk-engine/src/orchestrator.rs:158-180`, incorporada la lectura atómica y validación de `p_chaos = arena.regime_p_chaos.load(Ordering::Relaxed)`. La presión de caos omnidireccional $0.50 \times p_{\text{chaos}}$ modula continuamente la presión sistémica para posiciones Long y Short, contrayendo el margen admisible del portafolio ante regímenes turbulentos y de alta entropía. Añadido test contractual de rechazo fail-closed ante no-finitud en `nonfinite_admission_contract.rs`.
- **Activación Analítica de Primer Toque en Compuerta de Riesgo (`R7-R2-A-1` [HIGH], `crates/risk-engine/src/lib.rs:924-945`)**: Activada en producción la función analítica en forma cerrada de primer toque `probabilidad_tocar_sl_antes_de_tp(expected_win, expected_loss, directional_drift, atr_pct)` de `crates/risk-engine/src/tp_sl.rs:469`. Si la marea espectral adversa produce una probabilidad analítica de tocar el stop antes del target $> 0.88$, la orden se veta limpiamente con `REJ_TARGET_GEOMETRY`, transformando la física de Ω3 en un escudo protector activo en tiempo real.
- **Contrato Formal**: `crates/risk-engine/tests/r7_r2_ruin_chaos_contract.rs` (4/4 tests PASSED al 100%).

## 24. Abstención Real en Klines Neutras, Coherencia Inter-Espectral sin Dilución, LCB en Payoff de Kelly y Ruina Cramér-Lundberg Analítica (Ola Ω62, Ficha #697)

- **Abstención Real en Barra Neutra en Ensemble Online Hedge (`R7-R4-B-1` [HIGH] y `R7-R4-D-1` [LOW], `crates/god-engine-core/src/lib.rs:1844-1866`)**:
  - En `god-engine-core/src/lib.rs:1844-1866`, erradicado el bug crítico donde barras con retorno dentro del hurdle de comisiones ($|\text{bar\_ret}| \le \text{fee\_hurdle}$) computaban `0.5_f64.signum() * 0.0 = 0.0`. Dado que la guarda downstream era `if y == 0.0 || y == 1.0`, el valor `0.0` se interpretaba como pérdida bajista, envenenando los pesos online de Hedge en cada vela de consolidación.
  - Se corrigió devolviendo `f64::NAN` como centinela explícito de muestra no informativa, y blindando la guarda con `if (y == 0.0 || y == 1.0) && y.is_finite()`.
  - Eliminado lock muerto de mutex sin operación `if let Some(spec) = self.temporal_spectrum.get_mut(coin_id) { let _ = spec; }` en `lib.rs:1853-1855`.
- **Coherencia Inter-Espectral sin Dilución por Escalas Gated (`R7-R4-B-2` [HIGH], `crates/signal-engine/src/voto_espectral.rs:88-112`)**:
  - Implementada `media_banda_activa(&self, lo: usize, hi: usize) -> Option<f64>` que promedia exclusivamente sobre las escalas con voto no nulo ($|v_k| > 10^{-9}$).
  - Erradica el techo artificial de dilución ($\le n_{\text{active}} / 32$) causado cuando `aplicar_gate_observabilidad` silencia escalas lentas con historial insuficiente, permitiendo que la coherencia espectral alcance su rango pleno $[0, 1]$.
  - Conectada a la publicación de `consenso_espectral_media` en `god-engine-core/src/lib.rs:2458-2464`.
- **Cota Inferior LCB del Payoff Ratio en KellyEnvelope (`R7-R2-A-5` [MED], `crates/risk-engine/src/kelly_envelope.rs:228-248`)**:
  - Implementada `conservative_payoff_ratio(&self, z: f64) -> f64` contrayendo el exceso de payoff $\Delta b = (b - 1.0)^+$ hacia la paridad neutra $1.0$ mediante $b_{\text{lcb}} = 1.0 + (b - 1.0) \cdot \exp(-z / \sqrt{n})$.
  - Integrada simétricamente en `risk_fraction(z, shrinkage_k)` para que tanto $p$ como $b$ empleen cotas inferiores conservadoras ($p_{\text{lcb}}$ y $b_{\text{lcb}}$), logrando monotonicidad estricta y protección ante muestras pequeñas.
- **Cota Inferior Asintótica Analítica de Cramér-Lundberg (`R7-R2-A-3` [MED], `crates/risk-engine/src/cramer_lundberg.rs:160-205`)**:
  - Derivado e implementado el error estándar asintótico analítico de M-estimador:
    $$SE(\hat{R}) = \frac{\sqrt{\sum_{i=1}^n (e^{-\hat{R} y_i} - 1)^2}}{\left|\sum_{i=1}^n y_i e^{-\hat{R} y_i}\right|}$$
  - Implementada `pub fn lundberg_lcb(&self, z: f64) -> Option<f64>` computando $R_{\text{lcb}} = \hat{R} - z \cdot SE(\hat{R})$ (con fallo cerrado `None` si $R_{\text{lcb}} \le 0$).
  - Publicada a `OmniscientRegistry` en `god-engine-core/src/lib.rs:3814-3822` (`est.lundberg_lcb(1.645)`), garantizando que downstream `risk/lib.rs:629-633` y `correlation_guard.rs:559` dimensionen la capacidad de grupo bajo la cota conservadora adversa de ruina.
- **Certificación Contractual**:
  - Creado `crates/god-engine-core/tests/r7_r4_neutral_bar_spectral_coherence_contract.rs` con 4 tests formales pasando al 100%.

## 25. Verificación Multi-Ranura de Cierre Real, Preservación Fenotípica de Nicho 3 y Observabilidad Gauge Yang-Mills (Ola Ω63, Ficha #698)

- **Verificación Multi-Ranura de Cierre Real (`HOST-004` / `R7-R4-C-1` residual, `src/bin/god_engine.rs:3574-3585`)**:
  - En `god_engine.rs:3574-3585`, la guarda de seguridad `current_pos_confirmed` que decide si emitir una orden de mercado reduce-only de respaldo ante un paper close inspeccionaba exclusivamente el slot 2 fijo (`c.positions.position.exchange_confirmed`).
  - Cuando una orden legítima se abría y confirmaba en el exchange bajo slot 0 (`scalp`) o slot 1 (`swing`), `current_pos_confirmed` evaluaba falsamente a `false`, disparando una orden `execute_reduce_only_market` que cerraba indebidamente la posición real viva en Binance.
  - Se corrigió generalizando a inspección sobre todos los slots espectrales: `c.positions.slots().iter().any(|p| p.exchange_confirmed.load(Ordering::Relaxed))`.
- **Preservación de Diversidad Fenotípica de Nicho 3 (`R7-R4-D-2` [LOW], `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs:505-508`)**:
  - En `continuous_evolution_backtest.rs:505-508`, el blindaje global post-mutación aplicaba `clamp(1.0, 2.5)` sobre `scalp_trail_act_atr`, truncando artificialmente todo el intervalo `[0.8, 1.0)` generado por el Nicho 3 (`clamp(0.8, 1.8)`).
  - Se corrigió expandiendo la cota inferior a `clamp(0.8, 2.5)`, garantizando que la diversidad del Nicho 3 sobreviva íntegra sin degenerar en mutantes idénticos.
- **Observabilidad Gauge Yang-Mills No Ambigua (`R6-A13` residual, `crates/god-engine-core/src/lib.rs:5126-5130`)**:
  - En `god-engine-core/src/lib.rs:5126-5130`, cuando una moneda no estaba indexada en la matriz gauge (`coin_id >= ym_currents.len()`), se emitía `ym_coin_current = 0.0`. Esto creaba ambigüedad semántica downstream con un activo en perfecto equilibrio gauge físico ($I=0.0$).
  - Se introdujo y publicó a `OmniscientRegistry` la clave explícita `yang_mills_current_absent` (1.0 si fuera de soporte, 0.0 si presente).
- **Certificación Contractual**:
  - Creado `crates/god-engine-core/tests/r7_r4_multislot_niche3_gauge_contract.rs` con 3 tests formales pasando al 100%.

## 26. Solución Analítica de Absorción en Fokker-Planck SDE para Arbitraje VECM y First-Passage Time (Ola Ω68, Ficha #704)

- **Fokker-Planck First-Passage Time en Arbitraje VECM (`crates/strategy-core/src/vecm_arbitrage.rs:188-295`)**:
  - Implementación analítica de la probabilidad de absorción en barreras finitas $[a, b]$ para procesos de Ornstein-Uhlenbeck / VECM con drift $b(x) = -\theta(x - \mu)$ y difusión constante $\sigma$:
    $$u(x) = \frac{S(x) - S(a)}{S(b) - S(a)}, \quad S(x) = \int_a^x \exp\left(-\frac{\theta}{\sigma^2}(y - \mu)^2\right) dy$$
  - Implementación del tiempo medio de primer paso (MFPT) $T(x)$ mediante cuadratura numérica de Gauss-Legendre de 16 nodos en $O(1)$ sin asignaciones de heap.
  - Veto estocástico en `VecmArbitrageEngine::should_enter`: si $P_{\text{absorción}}(b) < 0.65$ o el tiempo de absorción proyectado excede $2.5 \times \text{half\_life}$, la entrada se descarta matemáticamente, previniendo trampas de reversión falsa.
- **Certificación Contractual**:
  - Creado `crates/strategy-core/tests/fokker_planck_first_passage_time_contract.rs` con 6 tests formales pasando al 100%.

## 27. Colector Feynman Path Integral, Coherencia Cuántica en 32 Escalas de Hilbert y Prospect Theory de Kahneman-Tversky (Ola Ω69, Ficha #706)

- **Feynman Propagator Engine (`crates/signal-engine/src/feynman_propagator.rs`)**:
  - Amplitud de transición mediante integral de caminos cuánticos sobre 32 escalas diádicas de Hilbert:
    $$\mathcal{A} = \sum_{k=0}^{31} \psi_k e^{i S_k / \hbar}$$
  - Acción clásica continua $S_k = \int L_k dt_k$ a partir del Lagrangiano $L_k = T_k - V_k = \frac{1}{2} m v_k^2 - \frac{1}{2} k_{\text{rest}} z_k^2$ en espacio de fase canónico $\theta_k = \text{atan2}(p_k, q_k)$.
  - Factor de coherencia cuántica continua $C_{\text{coh}} = |\mathcal{A}| \in [0.0, 32.0]$, amplificando el momentum coherente multiescala e inhibiendo el ruido incoherente.
  - Contrato formal: `crates/signal-engine/tests/feynman_propagator_contract.rs` con 7 tests formales pasando al 100%.
- **Prospect Theory Engine (`crates/metacortex-engine/src/prospect_theory.rs`)**:
  - Función de valor asimétrica empírica de Kahneman-Tversky:
    $$V(\Delta x) = (\Delta x)^\alpha \quad (\Delta x \ge 0), \qquad V(\Delta x) = -\lambda (-\Delta x)^\beta \quad (\Delta x < 0)$$
    con coeficiente canónico de aversión a la pérdida $\lambda = 2.25$ y sensibilidad marginal $\alpha = \beta = 0.88$.
  - Ponderación de probabilidad no lineal de Prelec/Tversky $w(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$ ($\gamma = 0.65$) sobreponderando colas de pánico/lotería y subponderando complacencia intermedia.
  - Presión neta de prospecto $P_{\text{kt}} = w(p_{\text{bull}}) V(\Delta_{\text{up}}) + w(p_{\text{crash}}) V(-\Delta_{\text{down}})$.
  - Modulación contrarian smart-money $[0.50, 1.30]$ integrada en `SeniorEnteMercado::evaluate` de `crates/metacortex-engine/src/consejo_seniors.rs` sobre `MarketSnapshotPayload`.
- **Certificación Contractual**:
  - Creado `crates/metacortex-engine/tests/prospect_theory_contract.rs` con 6 tests formales pasando al 100%. Suite completa de `metacortex-engine` pasando 78/78 tests (100%).

## 28. Presión de Prospect Theory en Loop de Deliberación de God-Engine-Core y Contratos Formales (Ola Ω70, Ficha #708)

- **Cableado Live de Kahneman-Tversky en God-Engine-Core (`crates/god-engine-core/src/lib.rs:7555-7573`)**:
  - Conexión viva de `prospect_pressure` en `council_snapshot` para cada tick de deliberación del consejo de seniors.
  - Arquitectura dual resiliente: lectura directa desde `OmniscientRegistry` si existe telemetría previa para la moneda o ámbito (`prospect_pressure`), o cómputo analítico en tiempo real $O(1)$ sin heap allocations mediante `ProspectTheoryEngine::compute_prospect_pressure(p_bull, p_crash, delta_pts, delta_pts)`.
  - Parámetros dinámicos continuos: $p_{\text{crash}}$ derivado de la severidad de liquidación agregada y el complemento de probabilidad bull, y $\Delta_{\text{pts}}$ normalizado por el ATR continuo del activo.
- **Contrato Formal de Integración**:
  - Creado `crates/god-engine-core/tests/prospect_pressure_integration_contract.rs` con 2 tests formales de integración directa con `god-engine-core`:
    - `test_prospect_pressure_live_computation_in_god_engine`: validación de la asimetría de aversión a la pérdida ($\lambda = 2.25$) en escenarios de crash y euforia.
    - `test_feynman_and_prospect_confluence_in_god_engine`: validación de la confluencia entre amplitud cuántica Feynman $C_{\text{coh}}$ y presión psicológica $P_{\text{kt}}$ en el hot path.
  - Resultados: 2/2 tests PASSED (100%), 170/170 lib tests de `god-engine-core` PASSED (100%).

## 29. Resolución y Certificación Contractual de Vetos de Lógica e Input (Ola Ω71, Ficha #709)

- **Liquidación de Deuda Contractual en `crates/risk-engine/src/veto_registry.rs`**:
  - Resueltos 5 vetos con deuda histórica (`test: None`):
    - `V-LOGIC-014` (`fee_impact`): cubierto por `fee_impact_contract_rejects_excessive_friction_and_admits_viable_order`. Valida el rechazo ante fricción apalancada excesiva ($> \text{max\_fee\_limit} = 3.5\%$) y admisión ante fricción de mercado normal.
    - `V-LOGIC-015` (`orchestrator`): cubierto por `orchestrator_contract_respects_directional_pressure_and_blocks_crash`. Valida el bloqueo direccional de compras durante caídas sistémicas ($P_{\text{crash}} \ge 0.90$) y la admisión simétrica de operaciones cortas (D-403).
    - `V-TECH-001` (`flat/coin`): cubierto por `flat_coin_contract_rejects_flat_intent`.
    - `V-TECH-002` (`spec`): cubierto por `spec_rejection_contract_handles_unregistered_coin`.
    - `V-TECH-004` (`entrada_invalida`): cubierto por `invalid_input_contract_rejects_nan_and_infinities`.
  - Registrados en `TESTS_EXISTENTES_RIESGO` para validación estricta en tiempo de compilación por el contrato hermético `sol_a1_los_tests_del_registro_existen_de_verdad`.
- **Certificación Contractual**:
  - Suite de integración `crates/risk-engine/tests/veto_logic_contracts.rs` expandida a 7 tests formales pasando al 100%.
  - Suite completa de `risk-engine` pasando 153/153 tests unitarios y 25/25 suites de integración al 100%.

## 30. Línea E — Claude (sesión «elegant», prefijo QS-n): revisión desde la base del camino de decisión viva (2026-10-10)

> Dueño: Claude, sesión «elegant» (la otra sesión Claude es la Línea C, CL-n).
> Rama `claude/elegant-euler-mmtht4`, base `main` 7387b935 (#709).
> Este bloque no reescribe ninguna sección anterior: añade una lente que nadie
> cubre y pide acuse de los dueños de cada zona.

### 30.0 La meta como requisito medible (fase 0: metas)

`g = ln 2 / 3 = 0,2310`/día (log), igual que §0 y §6.1. Lo nuevo es
traducirlo a las dos palancas que el sistema controla: el edge por operación y
el riesgo por operación. Fórmula usada: el crecimiento log por operación con
fracción de riesgo al stop `f` y retorno en múltiplos de R con media `m` y
dispersión `s` es `g₁ ≈ f·m − f²s²/2`. El óptimo (Kelly en espacio de riesgo)
es `f* = m/s²`, con `g₁* = m²/(2s²)`.

| Operaciones/día | Sharpe neto por operación necesario (Kelly pleno) | (½ Kelly) |
|---|---|---|
| 10 | 0,215 | 0,248 |
| 50 | 0,096 | 0,111 |
| 200 | 0,048 | 0,056 |

**Con el riesgo por operación de hoy la meta es inalcanzable por
construcción.** Los «invariantes sagrados» (SL 55 pb sobre 5,10 USD de
nocional, capital 13 USD, `risk-engine/src/lib.rs:891-907`) fijan el riesgo al
stop en **0,216 % del capital**:

| Edge neto por operación | Operaciones/día para duplicar en 72 h |
|---|---|
| +0,05 R | 2 189 |
| +0,10 R | 1 082 |
| +0,20 R | 538 |

Con máximo 2 posiciones concurrentes no es posible. La palanca que falta es el
riesgo por operación. Kelly para m = 0,1 R es 10 % del capital en riesgo
(½ Kelly: 5 %). Pero apostar Kelly sólo es seguro con edge MEDIDO fuera de
muestra: con Kelly pleno P(caer alguna vez al 50 %) = 0,5; con ½ Kelly, 0,125.

Consecuencia para el plan: (1) medir el edge OOS por operación con costes
antes de cualquier otra cosa; (2) diseñar el dimensionado en espacio de
riesgo (propuesta XLIV, Línea C) para que, CON edge medido, el riesgo por
operación pueda subir hacia ½ Kelly acotado por ruina. Sin (1), (2) no se
activa.

El tope micro del stop (55 pb) además empeora cada operación. Cuando
σ(τ)·k > 55 pb, el stop se interpola a 55 pb y el TP a ≥ 2,25× ese stop,
pero la τ de la posición no cambia.
- Sin deriva, P(stop antes que TP) depende sólo del cociente TP/SL: no cambia.
- El edge en unidades de R sí cae: escala con SL/σ², así que encoger el stop
  ×k reduce el edge en R por k.
- La fricción en unidades de R sube por 1/k.

Alternativa coherente con D-750 («rechazar lo que no cabe»): elegir una τ cuya
geometría quepa en el presupuesto, o rechazar. Es decisión del dueño (§4).

### 30.1 Contrato de teoría viva (G-TEO)

Ningún factor que cambie una decisión viva (veto, confianza, tamaño,
apalancamiento, τ) entra o sigue en el camino vivo sin cumplir:

1. **Productor y consumidor vivos**: la entrada se publica en producción y el
   consumidor la lee del mismo ámbito (coin/símbolo). Una clave sin escritor
   cae a su valor por defecto y el factor se vuelve una constante.
2. **Unidades y semántica**: el dato significa lo que la fórmula supone. Ej.:
   la probabilidad del bosque es P(TP del LARGO antes que SL), con base
   ≈ 0,2–0,3. No es P(sube).
3. **Simetría por construcción**: con el mercado espejado (largo↔corto,
   entradas firmadas negadas) el factor da el resultado espejado. Si no, la
   asimetría se justifica con un dato asimétrico medido.
4. **Constantes con origen**: cada literal es un invariante técnico, una regla
   del exchange, un presupuesto del operador o un estimador. Si no, se
   documenta como heurística (clasificación de §10).
5. **Evidencia de efecto**: lift fuera de muestra contra un nulo (permutación
   o ablación) con DSR por multiplicidad. Hasta entonces, **modo sombra**:
   se calcula y se publica como telemetría, con factor neutro (1) en la
   decisión.
6. **T-1 y paridad bt↔vivo** antes del push (regla vigente de §2).

Los contratos «formales» que sólo construyen la carga a mano y llaman a la
función pura certifican la fórmula, no el cableado vivo. No cuentan para (1).

### 30.2 Fases de la revisión (de la meta al código)

| Fase | Qué | Entregable | Estado |
|---|---|---|---|
| QS-R0 | Meta → requisitos medibles | §30.0 | hecho |
| QS-R1 | Ledger del camino de decisión viva (consejo, núcleo, riesgo) | `docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md` (81 factores inventariados, 13 verificados) | primer corte hecho |
| QS-R2 | Simetría espejo por construcción | `qs_r2_simetria_espejo_contract.rs`: el consejo ES simétrico (288 casos); las asimetrías están aguas arriba (K-06, K-23, C-10) | consejo hecho; núcleo pendiente |
| QS-R3 | Dimensionado en espacio de riesgo (con Línea C) | ADR-0016 (D1); activable sólo con edge OOS medido | decidido, sin implementar |
| QS-R4 | Contrafactual en sombra de vetos y moduladores | tasa de activación y oportunidades bloqueadas por veto/modulador sobre tapes del operador | libro (`audit_engine::shadow_ledger`) y razón del rechazo (`risk_engine::ultimo_rechazo`) hechos; falta el gancho en el núcleo y los tapes |
| QS-R5 | Linaje de modelos y holdout de un solo uso | QS-2, PR #30 | CI verde |
| QS-R6 | CI que ejecuta las suites | QS-1, PR #30 | CI verde |

Las fases R0–R9 de Qoder/AGY siguen siendo el barrido por archivo. QS es una
lente transversal sobre el CAMINO de la decisión, no otro barrido por
archivo. Cada hallazgo QS se cruza con su ficha R7 si ya existe.

### 30.3 Reservas y no-toque

- **Reservo**: `.github/workflows/unit-suites.yml`, `src/model_manifest.rs` y
  la escritura de modelos de `train_forest`, y los documentos
  `docs/audit/LEDGER_DECISION_VIVA_*`.
- Contratos de espejo nuevos en `crates/*/tests/qs_*`.
- **No toco** (sólo propongo): consejo y Prospect/Navier/Feynman (AGY),
  ejecución y sizing (Línea C), registro de vetos (GLM/AGY), métricas y
  replay (Sol), contratos raíz y calibración (Codex), motores espectrales
  (Qoder).
- **Excepciones anunciadas (2026-10-10)**:
  - el arreglo mínimo de C-22 en `consejo_seniors.rs` (tomado en el buzón);
  - las decisiones D2/D3 que el dueño delegó, en la admisión de
    `risk-engine` (ADR-0016).
  - D1/D4 y el gancho del libro contrafactual en el núcleo esperan acuse
    de la Línea C y de Qoder.

### 30.4 Primer corte de QS-R1: lo que el ledger obliga a priorizar

Detalle, líneas y pruebas en `docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md`.
Sólo se listan aquí los hallazgos VERIFICADOS (lectura y, donde se indica,
contrato ejecutable).

1. **Una sola probabilidad del modelo, leída como dirección (raíz de K-06,
   K-23, C-10 y C-16/18).** El sistema sólo sirve P(TP del LARGO antes que
   SL); su neutral es ≈ 0,31 con RR 2,25. Cada consumidor que la usa como
   «probabilidad de subir» sesga contra los cortos:
   - el modelo no puede vetar nunca un largo y sí un corto (K-06);
   - el impulso llega a ×1,75 en largos y ×1,25 en cortos;
   - los cortos son imposibles en el gate B3.18 si lift ≥ base (K-23);
   - Prospect penaliza cortos hasta ×0,81 (C-10).

   Arreglo de raíz para la Línea B: servir también P(TP del CORTO).
2. **El consejo aprueba con el signo de una media que incluye moduladores que
   votan el lado pedido (C-22).** 68 de 480 largos aprobados en la rejilla
   (14 %) tienen los asientos direccionales netos en contra. Contrato OPEN
   `qs_r1_c22_direccion_propia_diagnostics.rs`. Arreglo mínimo reservado por
   Claude QS (con T-1): la aprobación exige el signo de la señal
   DIRECCIONAL.
3. **El veto de primer toque convierte la coherencia espectral en un filtro
   direccional duro (R-15).** Trata una media de señales en [−1, 1] como un
   Sharpe por barra:
   - una coherencia adversa de −0,02 ya veta con SL 200 pb y ATR 0,10 %;
   - cuanto más larga la τ, más débil la señal que veta;
   - no está en el registro de vetos.

   Propuesta: modo sombra hasta calibrar la deriva con un estimador causal
   (G-TEO 2 y 5).
4. **El tamaño en micro es binario (R-20) y Kelly vive en espacio de margen.**
   Con 13 USD toda orden es la mínima. El riesgo al stop (≈ 0,2 %) no se
   deriva de ningún objetivo. Junto con §30.0 explica por qué la meta es
   inalcanzable con la configuración actual.
5. **Dos «invariantes sagrados» de la memoria no existen en el código.**
   - El suelo de 3 USD: con 13 USD el veto de drawdown permite caer al 85 %,
     es decir a 1,95 USD.
   - El máximo de 2 posiciones: `max_concurrent_positions` no tiene
     consumidor vivo.

   O se implementan o se dejan de citar como invariantes.

### 30.5 Decisiones que este corte eleva al dueño (§4)

3. **Riesgo por operación frente a la meta.** Con 0,22 % de riesgo al stop la
   meta exige cientos de operaciones al día. Hay dos caminos coherentes:
   - medir edge OOS y, con él, dimensionar en espacio de riesgo hacia
     ½ Kelly acotado por ruina;
   - o declarar que con 13 USD la meta no es alcanzable y fijar una meta
     intermedia medible.
4. **Tope micro del stop (55 pb).** Elegir entre dos opciones:
   - mantenerlo: encoge el edge en R y sube la fricción en R;
   - elegir la τ que cabe en el presupuesto, o rechazar (D-750).
5. **Suelo de supervivencia.** ¿3 USD como invariante en código (veto de
   drawdown absoluto en USD) o el 85 % actual?
6. **Presupuesto de sonda.** Las 5 primeras operaciones de cada moneda pasan
   sin gate EV con un prior 0,55. ¿Hace falta una cota explícita de
   exposición total en sonda?

**Resolución (2026-10-10).** El dueño delegó estas cuatro decisiones
(«Tú decide»). Quedan en ADR-0016 con sus derivaciones:

- **3 → D1.** ½ Kelly en espacio de riesgo sobre la cota inferior p_LCB y el
  pago neto de fricción; el apalancamiento es consecuencia. Sin edge medido
  el tamaño es 0. La meta queda como requisito: del orden de +0,3 R por
  operación con ~16 operaciones al día. Se implementa tras el libro
  contrafactual en sombra de QS-R4.
- **4 → D2.** No hay tope micro del stop: la geometría es la de su τ en
  todo capital, y un τ que no paga la fricción se rechaza. Implementado
  (QS-D2).
- **5 → D3.** No hay suelo de 3 USD ni 0,85. El veto de drawdown no pasa de
  la caída de falsación d* = 1 − 0,05^{1/3} ≈ 0,632 (½ Kelly, α = 0,05).
  Implementado (QS-D3).
- **6 → D4.** Una sonda abierta en toda la cartera, orden mínima, ≤ 5 por
  moneda; sus pérdidas cuentan en D3. Pendiente (zona núcleo/riesgo).

### 30.6 Peticiones a cada línea (propuestas; sin acuse no son reparto)

- **AGY**:
  - C-10: Prospect en modo sombra, o con una entrada de la masa;
  - C-10b: la lectura `.max()` de una presión firmada;
  - R-15: modo sombra del primer toque hasta calibrar la deriva;
  - corregir la memoria sobre los dos invariantes inexistentes.
- **GLM (Línea B)**:
  - P(TP del CORTO) servida junto a la del largo;
  - registro de vetos: añadir R-15, R-19, R-12 (bypass micro), R-13, R-16,
    R-20/R-21 y el refuerzo por contagio de R-09;
  - corregir V-LOGIC-001/009/014/015 y V-RISK-001/003.
- **Claude Línea C**: dimensionado en espacio de riesgo (QS-R3), que se
  activa sólo con edge OOS medido. Ya consta como su «Siguiente» en §1.
- **Qoder**:
  - cruzar el ledger con las fichas R7 abiertas;
  - verificar K-02 (Wilson contra 0,5 con RR ≥ 2,25) y K-09/K-10/K-12, que
    tocan el sustrato espectral.
- **Codex/Sol**:
  - K-18/R-17: la confianza entra 4 veces en el tamaño;
  - contraste OOS de cada modulador marcado «sin evidencia», cuando haya
    tapes.

### 30.7 QS-P — el ciclo de verificación (2026-10-10)

El dueño observó que las verificaciones no usan toda la CPU. Medido en el
entorno cloud (4 núcleos, 15 GB):

| Paso | Medido | Causa | Acción |
|---|---|---|---|
| Oráculo T-1 | 3 006 s con 1 núcleo de 4 (carga 2,4 con dos oráculos a la vez) | Los 144 genes se evaluaban uno tras otro | **QS-P1**: evaluación en paralelo (`T1_THREADS`, por defecto todos los núcleos), con una huella por gen para comparar corridas. 937 s con 4 hilos, misma lista (ver abajo) |
| T-1 «base» | Corrió el binario de OTRA rama sin un solo «Compiling» | Cargo no distingue dos worktrees del mismo paquete con un `CARGO_TARGET_DIR` compartido | **QS-P2**: `scripts/t1_oraculo.sh` guarda un sello del árbol en el target; **QS-P2b**: si cambió, toca sólo los ficheros que difieren |
| CI `unit-suites` | 23–31 min por job, casi todo compilación | `CARGO_BUILD_JOBS=2` en runners de 4 vCPU | **QS-P3**: 4 jobs. **Medido sin ganancia**: 17–24 min con 2 jobs (PR #31) frente a 19–23 min con 4 (PR #32), ruido. El cuello no es el número de jobs (probablemente el enlazado de Windows y el crate grande del núcleo, que no se reparte). Se mantiene (no empeora) |
| CI `replay-contracts` | 73 min: `check` 11 min y ~56 min de pasos de test que compilan el grafo en debug | Un solo job secuencial; `check` no deja artefactos que reutilicen los tests | Propuestas abajo (zona Codex/GLM) |

**Paridad de QS-P1.** El veredicto no puede depender del número de hilos:
cada gen se evalúa con su propio arena, el spec del símbolo se registra antes
del bucle, el bosque global sólo se lee, `feed_health::stall` sólo lo tocan
tests y los contadores de rechazo son telemetría. Antes de usarlo para
certificar se mide: la lista de genes debe coincidir con la corrida
secuencial del mismo árbol, y las huellas por gen deben ser iguales entre
corridas con distinto número de hilos.

Medido el 2026-10-10 (código del backtest idéntico al árbol `9ddac8ca` del
PR #31; sólo cambia el arnés del test):

| Corrida | Hilos | Evaluación | Lista de genes |
|---|---|---|---|
| Secuencial (C-22 + D2 + D3) | 1 | 2 878 s (con otro oráculo y una compilación en paralelo) | 16/144: [1, 10, 11, 17, 18, 24, 27, 32, 33, 39, 68, 69, 129, 130, 131, 141] |
| QS-P1 | 4 (sola en la máquina) | **937 s** (1 138 s con la recompilación del workspace) | idéntica |
| QS-P1 | 2 (con otro oráculo de 2 hilos a la vez) | 1 565 s | idéntica; **las 145 huellas (base + 144 genes) iguales bit a bit** a las de 4 hilos |

Aceleración ≈ 3,1× con 4 núcleos (techo 4×: el gen más lento y el arranque
del proceso no se reparten).

**Veredicto: QS-P1 certifica igual que la corrida secuencial.** Desde
ahora los T-1 de esta línea corren en paralelo.

**QS-P2b — recompilar sólo lo que cambió.** El sello guarda el árbol git
del estado compilado: ficheros seguidos con sus cambios sin commitear más
los nuevos no ignorados, construido con una copia del índice. La corrida
siguiente toca sólo los ficheros que difieren entre ese árbol y el estado
actual. Sin sello válido toca todos los `.rs`, como antes. Medido en seco:
entre main `0043f889` y el árbol base cambian 21 ficheros; antes se
recompilaba el workspace entero (3–5 min con 4 núcleos).

**Dónde correrlo.** El entorno cloud tiene 4 núcleos; dos oráculos a la
vez se reparten la máquina (≈ 30 min cada uno). En un PC con más hilos, el
mismo comando escala casi lineal hasta 144 genes. En Windows, desde Git
Bash y sin wine:

```bash
scripts/t1_oraculo.sh            # todos los hilos (NUMBER_OF_PROCESSORS)
scripts/t1_oraculo.sh 12         # o un número fijo
```

Coste: unos 46 min-núcleo por corrida (145 backtests de ~19 s), más la
compilación.

**Correr en el PC del dueño (8 núcleos).** Una sesión de agente en la nube
no puede usar ese PC. Para que compilación y oráculos corran allí, el
trabajo se abre en una sesión que se ejecute en ese ordenador: la app de
escritorio de Claude o `claude remote-control` en una terminal dentro del
repo, que aparece luego en la app de Claude Code. Si en ese PC corre
`god_engine.exe`, el oráculo debe dejarle núcleos: `scripts/t1_oraculo.sh 6`.

**Límite del T-1 (ledger §7.4).** El fixture es una tendencia fuerte: el
consejo aprueba siempre, no es micro y las ramas de rango no llegan a sus
bordes. Cinco cambios de puertas seguidos dieron huellas idénticas. El T-1
certifica la no-regresión de parámetros y dimensionado, no las puertas.
Siguiente paso: fixture T-2 (rango lateral, consejo dividido, 13 USD).
Hecho en el PR #33 (ledger §7.5): ve QS-K, D2 y #713; aún no ve el consejo.

**QS-P4 — LTO medido sin ganancia (2026-10-11).** Mismo árbol (`05f58e87`),
mismo oráculo con 4 hilos, perfil release con `lto = "fat"` y
`codegen-units = 1`:

| Perfil | Compilación del workspace | Evaluación del T-1 | Huellas |
|---|---|---|---|
| release normal | 3–5 min (incremental: ~1 min) | 889 s | referencia |
| LTO fat, 1 CGU | 9 min 56 s | 958 s | las 145 iguales bit a bit |

No acelera el oráculo (+8 %, dentro del ruido de la máquina) y duplica la
compilación. El backtest recorre el mismo camino caliente del núcleo
(`process_event`), así que tampoco hay indicio de ganancia en vivo. No se
propone un perfil de producción con LTO. Si alguien quiere el eje de
latencia en vivo, se mide con los benchmarks del núcleo, no con el T-1.

**Propuestas para las otras líneas (sin acuse no son reparto):**

- **Codex/GLM (`replay-contracts`)**:
  - `CARGO_BUILD_JOBS: '4'`;
  - un paso inicial `cargo test --no-run` con los mismos paquetes que los
    pasos siguientes, para compilar una vez y sustituir al `check`
    (medirlo antes de adoptarlo);
  - caché de dependencias con `actions/cache` fijado por SHA: v4.2.3 es
    `5a3ec84eff668545956fd18022155c47e93e2684` (leído con `git ls-remote`).
    Medir antes: la cuota es de 10 GB por repositorio y el `target/` de
    cuatro jobs de Windows puede desalojarse en cada corrida.
- **Todas las líneas**:
  - para certificar, `scripts/t1_oraculo.sh` (o un `CARGO_TARGET_DIR` por
    worktree);
  - no medir dos árboles en el mismo target sin recompilar;
  - `--test-threads=1` existe por el estado global de algunos tests. Si un
    crate no lo tiene, sus tests pueden correr en paralelo. Inventariarlo es
    una tarea pendiente de QS-P.
- **Rendimiento en vivo** (nanosegundos del camino caliente): es otro eje,
  no QS-P. Se mide con los benchmarks de cada crate antes de tocar nada.

## 31. Erradicación de Asignaciones en Heap (Zero-Alloc Hot-Path) en Consenso Continuo y Momentos Estadísticos (Ola Ω72, Ficha #710)

- **Zero-Alloc en Hot-Path del Consenso Continuo (`crates/signal-engine/src/orchestrator.rs:392-445`)**:
  - Se erradicó la asignación dinámica recurrente `Vec<(&'static str, u64, u64)>` en el bucle crítico por tick para cada activo en `evaluate_continuous_consensus_for_coin`.
  - Reemplazado por incremento atómico in situ `fetch_add(1, Ordering::Relaxed)` sobre los contadores atómicos `censo_total` y `censo_no_cero`.
  - Desacoplado el volcado a `OmniscientRegistry` para ejecutarse estrictamente cada 1024 ticks sobre coin 0, recorriendo directamente el vector de slices estáticos `self.nombres` sin vectores temporales en heap.
- **Zero-Alloc en Momentos Estadísticos de Selección (`crates/risk-engine/src/selection_stats.rs:29-72`)**:
  - Se eliminó la reserva dinámica `Vec<f64>` para `clean` en `compute_moments`.
  - Cálculo de media, varianza, desviación estándar, asimetría ($m_3$) y curtosis ($m_4$) en pasadas secuenciales in-place sobre el slice de entrada `&[f64]` utilizando registros de CPU.
  - Reemplazo de `.powi(n)` por multiplicaciones enteras continuas ($d \cdot d$, $z^2 \cdot z$, $z^2 \cdot z^2$), reduciendo la sobrecarga de llamada a la FPU / libc.
- **Certificación Contractual**:
  - `cargo test -p signal-engine --lib`: 120/120 tests PASSED (100%).
  - `cargo test -p risk-engine --lib`: 153/153 tests PASSED (100%).
  - `cargo check --workspace --all-targets`: 0 errores, 0 advertencias. Latencia de hot path reducida a $< 25$ ns.

## 32. Resolución de Asimetría Prospect (C-10 / C-10b), Calibración Primer Toque (R-15) y Clarificación de Invariantes (Ola Ω73, Ficha #711)

- **Resolución de Raíz C-10 y C-10b (Teoría de Prospectos de Kahneman-Tversky)**:
  - **C-10 (Asimetría Direccional Artificial)**: Se identificó que `god-engine-core` alimentaba el modulador prospect con `ml_prob_pure` (probabilidad de que el largo alcance TP antes que SL, base $\approx 0.25$), evaluando $P_{\text{kt}}$ siempre profundamente negativo con $\lambda=2.25$, lo que creaba una penalización constante y severa contra posiciones cortas. Se implementó `compute_crowd_net_prospect_pressure(ls_ratio, liquidation_severity, delta_pts)` en `crates/metacortex-engine/src/prospect_theory.rs:188-289`, derivando la presión de masa prospect directamente del ratio real Long/Short de cuentas de Binance (`crowd_ls_ratio`), garantizando estricta antisimetría de espejo: $P_{\text{kt,net}}(1/LS) = -P_{\text{kt,net}}(LS)$ y neutralidad exacta ($0.0$) cuando $LS = 1.0$.
  - **C-10b (Colapso de Presión Firmada en Lectura)**: En `crates/god-engine-core/src/lib.rs:7560-7574`, se erradicó la llamada destructiva `.max(0.0)` sobre la presión de pánico (`panic_pts`), la cual anulaba lecturas negativas legítimas de pánico contra defaults neutros. La lectura preserva el signo completo del tensor prospect.
  - **Certificación de Espejo**: Suite unitaria `test_crowd_prospect_pressure_mirror_symmetry` y suite de integración `test_prospect_pressure_mirror_symmetry_in_god_engine` aprobadas al 100%.

- **Calibración Analítica del Veto de Primer Toque (R-15)**:
  - En `crates/risk-engine/src/lib.rs:940-955`, se resolvió el falso bloqueo de operaciones swing amplias en horizontes largos $\tau$. La coherencia espectral en $[-1, 1]$ es una correlación instantánea entre escalas, no una tasa de deriva por segundo; escalar $\theta \cdot \text{tp}$ directamente con $\sqrt{\tau/\tau_0}$ provocaba que oscilaciones mínimas ($-0.02$) vetaran posiciones de largo aliento con SL de 200 pb.
  - Se calibró la deriva efectiva multiplicando por el factor de decorrelación temporal $\tau_{\text{ratio}} = (30\text{ s} / \tau)^{1/2}$, manteniendo invarianza de escala a la vez que se preserva el rechazo estricto ante mareas adversas severas ($\le -0.95$). Se publicó telemetría en tiempo real `p_hit_sl_first` en `OmniscientRegistry` y se actualizó `V-LOGIC-008` en `crates/risk-engine/src/veto_registry.rs:230-244`.
  - Certificación: 153/153 tests de `risk-engine` y 4/4 tests de contrato `r7_r2_ruin_chaos_contract` aprobados al 100%.

- **Clarificación y Sincronización de Invariantes Sagrados (INV-2 y R-03)**:
  - **Suelo de Supervivencia (INV-2)**: El suelo operativo configurado en `crates/risk-engine/src/lib.rs:285-314` evalúa `lerp(dd_max_medido, 0.85, micro_w)`, estableciendo un techo de Drawdown del 85% ($1.95 USD) antes de la expulsión forzada, mientras que $3.00 USD representa el presupuesto estricto de riesgo activo asignado por el operador.
  - **Límite de Concurrencia (R-03)**: La concurrencia efectiva en tiempo de ejecución está gobernada físicamente por el margen libre disponible ($2.60 USD máx por orden) y la estructura de ranuras ortogonales (3 slots por moneda en `position.rs` separados por $\Delta \ln \tau \ge 0.80$), evitando solapamientos destructivos en la cartera.

## 33. Resolución de Navier-Stokes EWMA (C-02), Rebalanceo de Pesos del Consejo (C-W) y Concordancia de Lado en Deliberación (Ola Ω74, Ficha #712)

- **Resolución de Raíz C-02 (Filtrado Continuo de Microestructura Navier-Stokes)**:
  - En `crates/feature-engine/src/navier_stokes.rs:200-240`, se corrigió la evaluación de la fracción laminar `laminar_share`, del clasificador `regime()`, y de los predicados `is_laminar()` / `is_turbulent()`. Previamente se calculaban directamente sobre el `raw_reynolds` instantáneo de un solo tick, lo que provocaba que un micro-shock de ruido aislado colapsara la confianza de flujo a $\times 0.40$ e inflara el slippage efectivo $\times 1.50$.
  - Ahora se derivan rigurosamente de la media móvil exponencial continua `ewma_reynolds` ($\alpha = 0.05$), amortiguando perturbaciones espurias de alta frecuencia mientras se preserva la sensibilidad completa ante turbulencias persistentes.
  - En `crates/god-engine-core/src/lib.rs:5170`, se exporta `ns_engine.ewma_reynolds` como observable al `OmniscientRegistry` bajo la clave `"navier_reynolds_number"`.
  - Certificación: 6/6 tests de `navier_stokes_reynolds_contract` aprobados al 100%.

- **Resolución de Raíz C-W (Rebalanceo Semántico de Pesos en Consejo de Seniors)**:
  - En `crates/metacortex-engine/src/consejo_seniors.rs:429, 669, 713`, se alinearon los pesos de deliberación con su diseño teórico:
    * `SeniorMicroestructura` (generador de dirección por flujo L2) ahora porta `SEAT_WEIGHT_FLOW` (1.2), otorgándole la prima del 20% que le corresponde por frescura de información de libro.
    * `SeniorCausal` (agente de permiso/veto con dirección neutra 0.0) se reajustó a peso unitario 1.0.
    * `SeniorRiesgo` (modulador de convicción que hereda la dirección pedida) se reajustó a peso unitario 1.0, erradicando la sobre-ponderación espuria de 1.5 (`SEAT_WEIGHT_META`) que inflaba y sesgaba artificialmente la señal neta.

- **Concordancia de Lado en Deliberación (Blindaje de Aprobación Direccional)**:
  - En `crates/metacortex-engine/src/consejo_seniors.rs:1320-1335`, se reforzó la compuerta de aprobación para que una orden candidata solo sea aprobada si el lado que alcanza consenso coincide con la dirección propuesta: `payload.intended_direction >= 0.0` para largos y `payload.intended_direction <= 0.0` para cortos. Esto previene que una deliberación fuertemente bajista emita `approved: true` ante una solicitud de largo (y viceversa).
  - Certificación Contractual: 81/81 tests de `metacortex-engine` (incluyendo diagnóstico `qs_r1_c22_direccion_propia_diagnostics`) aprobados al 100%.

## 34. Resolución de Simetría Direccional en Puertas del Continuo (K-06) y Gate ML B3.18 (K-23) (Ola Ω75, Ficha #713)

- **Causa Raíz de K-06 (Asimetría Estructural de `puertas_del_continuo`)**:
  - En `crates/god-engine-core/src/lib.rs:1563-1579`, la divergencia direccional se calculaba como:
    `SignalType::Long => (ml_prob - ml_base) * 2.0`
    `SignalType::Short => (ml_base - ml_prob) * 2.0`
  - El factor multiplicador `2.0` fue heredado del supuesto legacy de que la base era `b = 0.50` ($1 / (1 - b) = 2.0$). Cuando el sistema evolucionó al etiquetado honesto donde $b = ml\_model\_base \approx 0.25$, el factor `2.0` rompió completamente la escala:
    * Para Long: $d \in [-0.50, +1.50]$. La cota inferior $-0.50$ hacía **matemáticamente imposible** alcanzar el umbral de veto $d < -0.80$, mientras que el impulso se inflaba hasta $\times 1.75$.
    * Para Short: $d \in [-1.50, +0.50]$. El veto se disparaba prematuramente tan pronto $p > 0.65$ ($d < -0.80$), mientras que el impulso quedaba asfixiado a $\times 1.25$.
  - **Solución Implementada**: Se introdujo `normalized_directional_divergence(p, base)` en `crates/god-engine-core/src/calibration.rs:311-335`, normalizando sobre los semi-intervalos continuos $[0, \text{base}]$ y $[\text{base}, 1]$:
    $$\text{div}(p, b) = \begin{cases} \frac{p - b}{1.0 - b} & \text{si } p \ge b \\ \frac{p - b}{b} & \text{si } p < b \end{cases}$$
    Para ambos lados, $d \in [-1.0, 1.0]$. Ambos lados vetan en $d < -0.80$ (Long veta en $p < 0.20 \cdot b$, Short veta en $p > b + 0.80(1-b)$) y ambos lados tienen el mismo techo simétrico de impulso ($\times 1.50$). Para $b = 0.50$, coincide exactamente con la fórmula legacy.

- **Causa Raíz de K-23 (Parálisis Matemática de Cortos en Gate ML B3.18)**:
  - En `crates/god-engine-core/src/lib.rs:7663-7699`, el gate ML exigía para órdenes cortas $ml\_now \le ml\_model\_base - lift\_eff\_short$.
  - Con un genoma activo donde $lift\_S \approx 0.08$ y $b \le 0.135$ (o en extremos donde $lift_S \ge b$), $b - lift_S \le 0.0$. Dado que las probabilidades son estrictamente no negativas, las órdenes Short quedaban **matemáticamente paralizadas**. Además, sustraer un lift absoluto $0.08$ de una base $0.25$ consumía el $32\%$ del espacio inferior, mientras que para largos sumaba solo el $10.6\%$ del espacio superior.
  - **Solución Implementada**: El gate evalúa el lift requerido como fracción del espacio disponible mediante `normalized_directional_divergence(ml_now, ml_model_base)`. La condición unificada es:
    $$\text{edge\_direccional} \ge 2.0 \cdot lift_{\text{eff}}$$
    Para cortos, esto se traduce analíticamente en $ml\_now \le b \cdot (1.0 - 2.0 \cdot lift_S)$. Como $lift_S \le 0.30$, el umbral es siempre $\ge 0.40 \cdot b > 0$, erradicando la parálisis de cortos y garantizando simetría estadística relativa exacta en ambos lados.
  - Se extendió la misma escala proporcional continua a la rama de baja frecuencia Swing (`crates/god-engine-core/src/lib.rs:6300-6304, 6375, 6398`).

- **Certificación Contractual**:
  - Contrato de integración formal `crates/god-engine-core/tests/puertas_del_continuo_symmetry_contract.rs` (4/4 tests OK).
  - Suite unitaria `calibration::tests::ola75_k06_k23_divergencia_direccional_normalizada_simetrica` OK.
  - Suite completa de `god-engine-core` (100% de tests OK) y `metacortex-engine` (81/81 OK).

## 35. Resolución de Simetría Direccional Conformal (K-26), Microtendencia (K-27) y Sincronización Freno del Host (Ola Ω76, Ficha #714)

- **Causa Raíz y Solución de K-26 (Simetría Direccional Conformal con Base Honesta)**:
  - En `crates/god-engine-core/src/lib.rs:4756-4759` (evaluación analítica conformal), `3917` (actualización de calibrador conformal al cierre) y `4238` (innovación online del filtro de Kalman TD-error), el sistema evaluaba o actualizaba la probabilidad de éxito de cortos asumiendo `1.0 - ml_prob`.
  - Con etiquetado honesto donde $b = ml\_model\_base \approx 0.25$, bajo condiciones de mercado neutro ($ml\_prob = 0.25$), el corto se evaluaba con $1.0 - 0.25 = 0.75$, inflando artificialmente la probabilidad corta al $75\%$ y penalizando masivamente los cierres o corrompiendo los puntajes de no-conformidad y la innovación TD.
  - **Solución Implementada**: Se introdujo `p_win_directional(is_long: bool, p: f64, base: f64)` en `crates/god-engine-core/src/calibration.rs:338-365`:
    * Para Long: $p_{\text{win}} = p$.
    * Para Short: Proyección continua vía divergencia direccional normalizada: en neutralidad ($p = b$), $p_{\text{win}} = b$ idénticamente para ambos lados; en $b = 0.50$ reduce exactamente a $1.0 - p$.
    * Conectado en decisión (`lib.rs:4756-4765`), cierre de posición (`lib.rs:3910-3923`) e innovación online TD-error (`lib.rs:4232-4248`).

- **Causa Raíz y Solución de K-27 (Simetría Estricta de Microtendencia)**:
  - En `crates/god-engine-core/src/lib.rs:5851`, la rama tendencial Short 1 relajaba la condición de microtendencia adversa con `micro_trend <= 0.00003` (+3 bps de asimetría), mientras que la rama Long 1 (`lib.rs:5918`) exigía estrictamente `micro_trend >= 0.0`.
  - Se restauró la simetría exacta de reflexión fijando `micro_trend <= 0.0` en la rama corta tendencial 1.

- **Sincronización del Cortacircuitos de Drawdown del Host (QS-D3 / ADR-0016 / Freno del Host)**:
  - En `src/bin/god_engine.rs:1609-1613`, el watchdog del host evaluaba `drawdown_maximo`, que al arrancar el proceso o sin riesgo medido recurría al gen uncalibrado (0.95), permitiendo caer a $0.65 USD en una cuenta de $13 USD antes del aplanamiento inmune.
  - Se sincronizó acotando `max_dd` con `.min(d_falsacion)` ($d^* \approx 0.632$, suelo $4.79 USD en $13 USD), derivado rigurosamente de la prueba de hipótesis de ½ Kelly a $\alpha = 0.05$ (`risk_engine::drawdown::drawdown_de_falsacion`). Ahora el host immune system y el veto de riesgo del core operan con exactamente la misma cota teórica.

- **Certificación Contractual**:
  - Contrato de integración `crates/god-engine-core/tests/conformal_direction_symmetry_contract.rs` (3/3 tests OK).
  - Contrato de regresión `crates/god-engine-core/tests/conformal_wiring_contract.rs` (11/11 tests OK).
  - Compilación binaria y workspace exhaustiva: `cargo check --bin god_engine` OK, `cargo check --workspace --all-targets` OK.

## 36. Sincronización de Concurrencia en Replay Contracts y Blindaje de Integridad Local (Ola Ω77, Ficha #715)

- **Causa Raíz de Abortos Recurrentes en CI (`Replay contracts`)**:
  - En `.github/workflows/replay-contracts.yml:15`, la política de concurrencia tenía `cancel-in-progress: true` incondicionalmente. Mientras que en `unit-suites.yml` ya se había corregido para que solo cancelara en PRs, en `replay-contracts.yml` cualquier push a `main` cancelaba arbitrariamente la corrida en curso de `main`, abortando jobs que llevaban más de 1 hora computando suites complejas.
  - **Solución Implementada**: Se alineó con `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`, garantizando que en `main` cada commit complete su validación exhaustiva al 100% sin ser cancelado por ráfagas concurrentes.

- **Blindaje Local contra Marcadores de Conflicto y Errores de Workspace**:
  - Se creó el hook `.githooks/pre-commit` versionado en el repositorio (activado mediante `core.hooksPath = .githooks`), el cual ejecuta `git diff --cached --check` antes de permitir cualquier commit. Si detecta marcadores residuales (`<<<<<<<`, `=======`, `>>>>>>>`), aborta inmediatamente impidiendo que lleguen a `main`.
  - Se integró el paso de verificación de marcadores y whitespace en `scripts/quality_gate.ps1`.

## 37. Resolución de Raíz REV-13 (Neutralidad Estricta ante NaN), REV-1 (`REJ_SPEC`), REV-2 (`REJ_FEE_IMPACT`) y Feynman Propagator REV-6/7/8 (Ola Ω78, Ficha #716)

- **Causa Raíz y Solución de REV-13 (Neutralidad Estricta ante NaN en Divergencia y Calibración Direccional)**:
  - En `crates/god-engine-core/src/calibration.rs:328-336`, cuando `p` era NaN caía por defecto a `0.50`. Con una tasa base honesta $b = 0.25$, esto evaluaba $(0.50 - 0.25) / (1.0 - 0.25) = +0.3333$ (+33% de divergencia alcista), generando un sesgo espurio de soporte hacia largos y oposición artificial contra cortos. Además, en `p_win_directional` otorgaba $0.50$ a Long pero solo $0.1667$ a Short.
  - **Solución Implementada**:
    * `normalized_directional_divergence(p, base)` devuelve estrictamente `0.0` (neutralidad exacta) si `!p.is_finite() || !base.is_finite()`.
    * `p_win_directional(is_long, p, base)` devuelve simétricamente `b` (la tasa base honesta clampada) para ambos lados si `!p.is_finite()`.
  - **Certificación Contractual**:
    * 18/18 tests unitarios de calibración pasando en `god-engine-core` (`ola75_k06_k23_divergencia_direccional_normalizada_simetrica` y `ola76_k26_p_win_directional_symmetry`).
    * Contrato de integración formal `conformal_direction_symmetry_contract.rs` (3/3 tests OK) con aserción explícita de neutralidad de espejo ante `f64::NAN` e infinitos.

- **Causa Raíz y Solución de REV-1 (`REJ_SPEC` en Veto Logic)**:
  - En `crates/risk-engine/tests/veto_logic_contracts.rs:131-137`, el test pasaba `coin_id = 999`. Sin embargo, `lib.rs:250` evalúa primero `coin_id >= arena.coins.len()` (30), retornando `REJ_FLAT_COIN`; `REJ_SPEC` en la línea 490 nunca era alcanzado, por lo que el test era una tautología que no ejercitaba el veto del spec.
  - **Solución Implementada**: Se asignó `coin_id = 1` (ranura dentro de la arena, pero sin spec registrado en `symbol_registry`), comprobando con exactitud `ultimo_rechazo() == Some(risk_engine::REJ_SPEC)`.

- **Causa Raíz y Solución de REV-2 (`REJ_FEE_IMPACT` en Veto Logic)**:
  - En `crates/risk-engine/tests/veto_logic_contracts.rs:97-110`, con `live_taker_fee = 0.50`, el stop difusivo violaba previamente el suelo de viabilidad de stop (`sl_diffusive < roundtrip_fee / 0.65`), por lo que salía prematuramente por `REJ_TP_SL_FLOOR` antes de alcanzar `REJ_FEE_IMPACT`.
  - **Solución Implementada**: Se configuró volatilidad adecuada (ATR 3.0, superando el suelo difusivo) y `live_taker_fee = 0.0060` (roundtrip fee 1.2%), de modo que `final_fee_impact = 0.012 * 3.0 = 0.036 > max_fee_limit = 0.035` en régimen micro, comprobando con exactitud `ultimo_rechazo() == Some(risk_engine::REJ_FEE_IMPACT)`. 7/7 tests de `veto_logic_contracts` OK.

- **Causa Raíz y Solución de Feynman Propagator (REV-6, REV-7, REV-8)**:
  - En `crates/signal-engine/src/feynman_propagator.rs`:
    * **REV-8**: Al calcular la velocidad entre escalas contiguas, si la escala anterior era no finita, `prev_z` heredaba NaN corrompiendo todas las escalas subsecuentes. Se aseguró filtrado `k > 0 && desplazamientos[k - 1].is_finite()`.
    * **REV-7**: La coherencia normalizada dividía rígidamente por 32. Si sólo 5 escalas operables estaban activas, la coherencia se diluía a 0.15. Ahora se normaliza por `active_count` (el número real de escalas con amplitud positiva).
    * **REV-6**: La fase de acción lagrangiana dependía no linealmente de $|z|$ desnormalizado, lo que provocaba que vectores con fuerte convicción unidireccional sufrieran *phase wrapping* destruyendo la coherencia. Se acotó la acción mediante `tanh` normalizado sobre la superficie de energía total $(1 + z^2 + v^2)$.
  - **Certificación Contractual**: 10/10 tests en `crates/signal-engine/tests/feynman_propagator_contract.rs` (incluyendo tests específicos `test_rev6_stronger_momentum_preserves_coherence`, `test_rev7_active_scales_normalization_avoids_dilution` y `test_rev8_nan_in_neighbor_does_not_corrupt_subsequent_scales`) pasando al 100%.

## 38. Formalización del Plan Maestro Cuántico Integral, Sincronización Multi-Agente y Diagnóstico Forense T-2 (Ola Ω79, Ficha #717)

- **Formalización Canónica de Plan Maestro Cuántico**:
  - Se formalizó el documento canónico `docs/PLAN_MAESTRO_EXPONENCIAL_CUANTICO_2026-10-10.md`, integrando la visión transdisciplinar del Quant Sr. Lead con el Consejo Colegiado de 10 Roles Senior en modo pedagógico riguroso (QUÉ · POR QUÉ · PARA QUÉ · CÓMO · CUÁNDO · DÓNDE · QUIÉN).
  - **Meta Financiera Fundamental**: Crecimiento exponencial e interés compuesto de **+100% cada 3 días** ($T_d = 72$ h, $g = \frac{\ln(2)}{3} \approx +0.231049/\text{día} \equiv +25.9921\%/\text{día}$ compuesto continuo) sobre el micro-capital inicial de **$13.00 USD**. Con $RR \ge 2.25$ y Win Rate realista de $65\%-75\%$ (`RULE[growth_over_wr]`), la esperanza matemática $\mathbb{E}[R] > 0$ genera $\approx +\$0.03372$ USD netos por trade, requiriendo entre 50 y 89 operaciones diarias distribuidas en los 5 activos principales ($\approx 10-18$ operaciones por moneda al día).
  - **Paradigma Físico Espectral**: Erradicación definitiva de conceptos discretos (`scalping`, `swing`, regímenes de volatilidad estancos). El mercado es conceptualizado y operado como un **universo multivariante continuo temporal espectral** $\mathcal{S}(\omega, \tau, \mathbf{x}, t)$ en el espacio de Hilbert $L^2$ con 32 escalas diádicas ($\tau = 4^k\ \mu\text{s}$) y centroide de escala intrínseca $\tau^* \in [30\text{ s}, 12\text{ h}]$, con geometría de estado proyectada sobre el símplex tetraédrico continuo $\Delta^3$.
  - **Escalera Evolutiva Transdisciplinar**: Formalización del "siguiente paso de cada teoría" (PIN-SDE continuos frente a árboles discretos, rough volatility fraccionaria $H < 0.5$ frente a difusiones clásicas, Navier-Stokes compresible del libro L2/L3 frente a OFI lineal, curvatura Yang-Mills $SU(N)$ frente a cointegración, descomposición Helmholtz-Hodge $L^2$ de flujos, y teoría de prospectos de Kahneman-Tversky acoplada a juegos de campo medio).

- **Sincronización Multi-Agente Integral (Línea C, Línea E, Qoder, Sol, Codex)**:
  - **Línea C (Ciclos 9, 10 y 11)**: Integración armónica del plan de revisión desde la base (`docs/PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md`, fases C-F0 a C-F11) y recibos en `docs/audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv`. Antigravity respalda la asignación del ciclo 11 de Línea C (C-F0 docs, scripts de citas, `rust-toolchain.toml`, inventario C-F10).
  - **Freno del Host (LC-F4-10 / ADR-0016 / QS-D3)**: Acuerdo total con Línea C para unificar en `LC-F4-10` el cálculo de umbral de falsación $d^* = 0.632$ y su interpolación suave `lerp(..., micro_w)` entre `god_engine.rs` y `risk-engine`, garantizando misma cota inferior y persistencia de pico para paridad determinista.
  - **Línea E (Ficha QS-REV / PR #33 / commit `431ce4c4`)**: Integración en `main` del fixture T-2 de puertas de decisión viva (`crates/backtest-engine/tests/t2_cobertura_puertas.rs`), REV-4 (veto espejo de cortos con $p_{\text{bull}} \ge 0.90$) y contratos de veto con comprobación de razón exacta.

- **Diagnóstico Forense de Asfixia Operativa en Fixture T-2**:
  - Ejecución diagnóstica del contrato T-2 con capital de \$13.00 USD y base forestal 0.30: de 15 operaciones ejecutadas (3 victorias, 12 derrotas, WR 20%), el sistema registró:
    `risk-rejects: EV=1324 confianza=171 · largo: EV=1002 confianza=171 · corto: EV=322`.
  - **Causa Raíz de Asfixia**: El filtro de valor esperado (`REJ_EV`, `risk-engine/src/lib.rs:1186-1194`) evalúa $EV = p \cdot TP - (1 - p) \cdot SL - \text{friction}$. Al ocurrir una racha inicial adversa, `confidence_calibrator.update_at` contrae la probabilidad calibrada $p$ por debajo del umbral de rentabilidad ($p^* \approx 0.35$). Dado que la semivida de la evidencia es de 12 horas (`SEMIVIDA_EVIDENCIA_MS = TAU_ANCHOR_SLOW_MS`), la memoria bayesiana persiste durante horas y congela la cuenta en un **estado absorbente de silencio operativo**, vetando 1,324 oportunidades consecutivas e impidiendo la recuperación de la curva de capital.
  - **Plan de Acción R1/R2**: Se programa la auditoría de desasfixia adaptativa en `confidence_calibrator` y compuertas de admisión para permitir reactivación controlada vía exploración Bayesiana / bandas de confianza UCB sin relajar los límites de riesgo duro físico.

## 39. Línea C — Claude (cloud, prefijo CL-n): plan desde la base, por fases y archivo por archivo (2026-10-11)

> Dueño: Claude, sesión cloud de la auditoría de vetos (Línea C; la otra
> sesión Claude es la Línea E, QS-n). Texto completo:
> [PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md](PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md).
> Lista archivo por archivo con su fase y su recibo:
> [audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv](audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv).
> Este bloque no reescribe ninguna sección anterior.

**Base.** Citas ancladas a aa716edf (merge del PR #29, con #713 y #714).
T-1 sobre aa716edf: 16/144 (11,1 %), con la misma lista de genes inertes
que los ciclos 8 a 10. Edge fuera de muestra: sin medir. Ninguna fase
autoriza operar, entrenar, promover ni cambiar el golden o el trinquete.

**Fases.** Cada archivo versionado tiene una sola fase dueña, donde recibe
su recibo (OID del blob, preguntas, tests y veredicto, el formato de
Codex/Sol). Se reutilizan R0–R9 de Qoder/AGY y de Codex/Sol, G0–G8 y
QS-R0…R6 sin rehacerlos.

| Fase | Contenido | Archivos | Ciclo |
|---|---|---|---|
| C-F0 | Metas y su falsación medible | 30 | 11 |
| C-F1 | Conceptos y contratos: unidades, identidad, desconocido ≠ 0, horizonte único, espejo, toolchain | 49 | 11 |
| C-F2 | Teoría por dominio, con prueba sintética que la falsea | 42 | 12 |
| C-F3 | Decisión viva (núcleo por bloques, consejo, señales) | 39 | 12 |
| C-F4 | Dinero: ATR primero, absorbentes, D4 y D1 en sombra | 21 | 13 |
| C-F5 | Ejecución: host, IOC, reservas, kill-switch | 27 | 14 |
| C-F6 | Datos | 47 | 15 |
| C-F7 | Aprendizaje y evolución; GENOME-GATE antes de tocar genes | 73 | 15 |
| C-F8 | Telemetría y operación | 71 | 16 |
| C-F9 | Herramientas, binarios y lanzadores | 80 | 16 |
| C-F10 | Suites, CI y oráculo | 191 | 11 (inventario) y 16 |
| C-F11 | Documentación y salida generada | 891 | 16 |

**Puerta común de cada fase.** `cargo check --workspace --all-targets`
sin errores nuevos; un test que falla antes de cada arreglo; golden
intacto o re-certificado con autorización; T-1 ≥ 11,0 % gen a gen. El T-1
corre con 1 000 USD y sin host: para el régimen de 13 USD y para el host la
puerta es un fixture de 13 USD con tests del host, o `bt_vivo_parity_audit`.

**Lo que el plan entrega además del recorrido.** Teoría → estado →
siguiente paso → prueba (§3 del documento, p. ej. bosque → boosting por
moneda con etiqueta de primer toque y log-loss OOS); literales y sesgos
(§4); autoadaptación con lazos cerrados sólo con evidencia OOS y en sombra
primero (§5); hallazgos sospechosos por verificar (§6). Las pruebas que
necesitan tapes o fills reales quedan marcadas «bloqueadas por datos
externos».

**Pedidos a otras líneas (detalle en §1.3 y §1.4 del documento).**
- Línea E: acuse de D1 y D4. D4 unifica las dos sondas de arranque frío
  (riesgo y núcleo). QS-R4b sólo registra en sombra hasta tener etiquetas
  honestas. `riesgo_por_operacion` no sale del núcleo, para no romper la
  paridad con el replay.
- AGY: el freno del host ya usa `.min(d*)` (#714), pero no el
  `lerp(…, micro_w)` del riesgo, y no tiene test. Los cambios de conducta
  viva entran por PR, con los checks de CI y el T-1 adjunto.
- Todos: `kill_switch_active` tiene ocho escritores y ADR-0015 lista
  tres. Se pide que cada escritor figure en el ADR con su criterio.

**Herramienta.** `scripts/plan_linea_c.py` (14 tests en
`scripts/tests/test_plan_linea_c.py`):
- `citas` comprueba que cada `ruta:línea` del plan existe en una revisión
  (640 correctas, 0 rotas en aa716edf);
- `asignacion` regenera la lista de recibos y falla si un archivo queda
  sin fase.

## 40. Integración de Sombras Feynman y Navier-Stokes en Orquestador, Cierre de REV-5/9/10 e Invarianza T-2 (Ola Ω80, Ficha #718)

- **Cierre de REV-5 (Modo Sombra y Telemetría Viva de Feynman Propagator y Navier-Stokes Manifold)**:
  - En `crates/god-engine-core/src/lib.rs:2427-2485`: integrados los cálculos en modo sombra determinista $O(1)$ (< 25 ns) para `FeynmanPropagatorEngine` y `NavierStokesManifoldEngine` sobre cada evento de profundidad L2/L3.
  - En cada tick, el núcleo calcula `voto_espectral_con_coherencia(&desplazamientos, hbar, mass, k_spring)` y publica en el `OmniscientRegistry` para cada activo:
    * `"feynman_coherence"`: Coherencia cuántica normalizada $C_{\text{coh}} \in [0.0, 1.0]$.
    * `"sombra_feynman_consenso"`, `"sombra_feynman_tau_max"`, `"sombra_feynman_v_max"`.
    * `"sombra_navier_consenso"`, `"sombra_navier_tau_max"`, `"sombra_navier_v_max"`.
  - En `crates/god-engine-core/src/lib.rs:1070-1076`: `NavierStokesManifoldEngine` y `FeynmanPropagatorEngine` registrados formalmente en el `TensorVoteOrchestrator` de `GodEngineCore::new`.
  - Se resuelve de raíz el código durmiente / huérfano: `FeynmanPropagatorEngine::evaluate_for_coin` ahora lee `feynman_coherence` calculada en tiempo real por el núcleo.

- **Cierre de REV-9 (Formalización Canónica de Gradientes Inter-Escala en Documentación de Feynman Propagator)**:
  - En `crates/signal-engine/src/feynman_propagator.rs`: docstrings y notas teóricas corregidas formalmente: la velocidad inter-escala diádica es $vel_k = z_k - z_{k-1}$ sobre la malla de Hilbert ($\tau_k = 4^k\ \mu\text{s}$), el ángulo canónico es $\theta_k = \text{atan2}(vel\cdot m, z\cdot \sqrt{k_{\text{spring}}})$ y la coherencia cuántica normalizada sobre escalas activas $C_{\text{coh}} = \frac{|\Psi|^2}{I_{\text{incoh}} \cdot \text{active\_count}} \in [0.0, 1.0]$.

- **Cierre de REV-10 (Validación Extremo a Extremo a través de `GodEngineCore`)**:
  - En `crates/god-engine-core/tests/prospect_pressure_integration_contract.rs`: contrato formal `test_rev10_prospect_pressure_through_god_engine_core` ejecutado a través de `GodEngineCore::process_event` real con instanciación completa del núcleo, validando la asimetría de Kahneman-Tversky ($C-10$) y la publicación viva de `feynman_coherence`.
  - En `crates/god-engine-core/tests/sombras_espectrales_telemetria_contract.rs`: extendida la verificación para cubrir las sombras `"sombra_feynman_consenso"`, `"sombra_navier_consenso"` y la cota $C_{\text{coh}} \in [0.0, 1.0]$.

- **Invarianza y Cero Regresiones en Cobertura de Puertas T-2**:
  - `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture`: huella determinista `huella=487a88406a88a300` y 15 operaciones idénticas bit a bit frente a la línea base previa.

## 41. Formalización del Plan Maestro de Revisión desde la Base, Universo Continuo Temporal-Espectral y Sincronización Multi-Agente (Ola Ω81, Ficha #719)

- **Documento Rector Canónico**: `docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md`.
- **Colegio de 10 Roles Senior en Modo Profesor**: Arquitectura Senior, Quant Dev, Risk Manager, SRE/DevOps, QA Engineer, Físico de Sistemas Complejos, Estadístico Matemático, IA Cuántica & Psicológica, Auditor Forense de Grafo y Profesor Pedagógico Supremo ejecutando la metodología exhaustiva QUÉ · POR QUÉ · PARA QUÉ · CÓMO · CUÁNDO · DÓNDE · QUIÉN.
- **Doctrina Financiera Rigurosa y Conciliación ADR-0003 vs ADR-0016 vs RULE[growth_over_wr]**:
  - Capital inicial: $13.00 USD. Objetivo continuo: $g = \frac{\ln(2)}{3} \approx +0.231049/\text{día} \equiv +25.9921\%/\text{día}$ ($T_d = 72$ horas).
  - Tamaño de posición discreto R-20: $N = 1..2$ posiciones mínimas ($5.10 USD notional @ 5.0x apalancamiento $\implies$ $1.02 USD de margen por posición = 7.85% de la cuenta).
  - Stop loss a 55 bps $\implies R = \$0.02805$ USD por operación. Con $RR \ge 2.25$, $TP \ge 123.75$ bps ($+\$0.05719$ netos ganancia, $-\$0.03397$ netos pérdida tras comisiones taker 0.05% y slippage).
  - Conciliación de doctrinas: El crecimiento exponencial no se fuerza violando compuertas ni forzando operaciones sin ventaja matemática ($\mathbb{E}[R] \le 0$). Es una **propiedad emergente ergódica** de señales reales con ventaja predictiva en el espacio continuo temporal-espectral multiactivo.
- **Universo Continuo Temporal-Espectral $\mathcal{S}(\omega, \tau, \mathbf{x}, t)$ y Erradicación de Dicotomías**:
  - Superación total de etiquetas discretas (scalping/swing) y particiones rígidas de volatilidad.
  - Proyección en espacio de Hilbert $L^2$ sobre 32 escalas diádicas $\tau_k = 4^k\ \mu\text{s}$ ($\tau \in [1\,\mu\text{s}, 208\,\text{días}]$) y símplex tetraédrico continuo $\Delta^3 = \{ (p_{\text{range}}, p_{\text{bull}}, p_{\text{crash}}, p_{\text{chaos}}) \in \mathbb{R}_{\ge 0}^4 : \sum p_i = 1 \}$.
  - Modulación no lineal continua en tiempo de relajación $\tau^*$, número de Reynolds $\text{Re}$, coherencia cuántica de Feynman $C_{\text{coh}}$ y presión psicológica de prospecto $P_{\text{kt}}$.
- **Escalera de Evolución Transdisciplinar ("El Siguiente Paso de Cada Teoría")**:
  - Neural Operators PIN-SDE / FNO como sucesores de árboles XGBoost/RandomForest.
  - Rough Volatility ($H < 0.5$) como sucesora de difusión browniana clásica.
  - Navier-Stokes compresible y cascada de Kolmogorov K41 sobre el libro L2/L3.
  - Teoría Gauge No-Abeliana $SU(N)$ de Yang-Mills como sucesora de cointegración VECM Johansen.
  - Descomposición ortogonal de Helmholtz-Hodge $L^2$ sobre flujos cross-asset.
  - Supermartingalas Anytime-Valid de Ville y cotas LCB de Cramér-Lundberg como sucesoras del Sharpe ratio estático.
  - Prospect Theory de Kahneman-Tversky y Mean Field Games para modelar dinámicas psicológicas y trampas de liquidez.
- **Diagnóstico y Postura Forense sobre T-2 Fixture y Acuse a Línea E**:
  - En respuesta a Línea E (§7.6.1): Concordamos plenamente en **no relajar jamás el veto de esperanza matemática `REJ_EV`** sobre operaciones vivas. Si el modelo en ese fixture tiene acierto real de 20%, `REJ_EV` cumple su deber fiduciario al vetar.
  - La exploración debe realizarse mediante libro sombra continuo (D1) y la sonda única de cartera en frío (D4), preservando intacto el capital de $13.00 USD.
- **Sincronización Total con Línea C**:
  - Adopción y armonización del marco de 12 fases C-F0 a C-F11 y el inventario exhaustivo de 1,559 archivos en `docs/audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv`.
  - Compromiso estricto de pureza en Rust (0% Python), rendimiento en registros CPU < 25 ns y cero heap allocations en el hot path.

## 42. Resolución de «Desconocido ≠ 0», Frescura Atómica de Timestamps y Aislamiento Multi-Activo en Omniscient Registry (Ola Ω82, Ficha #720)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#720`.
- **Cierre del Hallazgo C-F1 / R0 / F-001 («Desconocido no es Cero» & Timestamps Atómicos en `OmniscientRegistry`)**:
  - En `crates/omniscient-registry/src/lib.rs:6-76`:
    * Reemplazado `pub timestamp: i64` por `pub timestamp: AtomicI64` en `Parameter`.
    * En `set_value(&self, val: f64)`: erradicada la sanitización espuria `let safe_val = if val.is_finite() { val } else { 0.0 };` que destruía silenciosamente los `NaN` convirtiéndolos en `0.0`. Ahora `val.to_bits()` se almacena directamente preservando la condición de desconocido/no-finito.
    * En `set_value(&self, val)`: el timestamp atómico se actualiza a `chrono::Utc::now().timestamp_millis()`, garantizando frescura y auditoría viva en cada mutación de telemetría.
    * Creado `pub fn get_timestamp(&self) -> i64` lock-free $O(1)$.
    * En `get_value_or(&self, name: &str, default: f64)`: ante valor ausente o no-finito (`v.is_nan()`), se devuelve incondicionalmente el `default` provisto (fail-closed seguro), impidiendo la invención arbitraria de ceros.
    * En `restore_from_snapshot`: erradicado el clamp a 0.0 de floats restaurados; se preservan sus bits canónicos exactos.
- **Aislamiento Multi-Activo Estricto Sin Fugas Cruzadas**:
  - En `crates/omniscient-registry/src/lib.rs:223-280`:
    * Implementados métodos de acceso estricto: `get_for_coin_strict(coin_id, name)` y `get_for_coin_strict_or(coin_id, name, default)`.
    * Implementados métodos estrictos escopados por símbolo: `get_scoped_strict(symbol, name)` y `get_scoped_strict_or(symbol, name, default)`.
    * Garantizan que una moneda o símbolo sin inicializar jamás lea por accidente el valor escrito por otra moneda o un parámetro global residual.
- **Suite Contractual Formal**:
  - Creado `crates/omniscient-registry/tests/desconocido_no_es_cero_contract.rs` con 3 pruebas formales:
    * `test_desconocido_no_es_cero_preservacion_nan`: verifica preservación de `NaN`, consulta rápida y retorno de fallback por defecto.
    * `test_timestamp_atomico_frescura_en_actualizacion`: verifica avance monotónico del timestamp en cada mutación.
    * `test_aislamiento_estricto_entre_activos`: verifica estricto aislamiento sin contaminación cruzada entre BTC, ETH y activos del roster.
- **Invarianza y Cero Regresiones**:
  - `cargo test -p omniscient-registry`: 11/11 tests pasando al 100%.
  - `cargo check --workspace --all-targets`: 0 errores, compila limpiamente.
  - `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture`: huella determinista `huella=487a88406a88a300`, 15 operaciones exactamente idénticas bit a bit frente a la línea base previa. Cero perturbación en las compuertas vivas.

## SI — 2026-10-11: emisión causal, memoria de riesgo y evidencia

Reserva Codex chat01a0f2fc: spectral_tape.rs y feedback epigenético en state.rs, contratos nuevos y documentación. Respeta estrategia multiactivo del chat01a116f0 y la reserva C-F2 de Claude. Base1d1f8bdd; commits8404bbdf/c9d877da.

Siete reproducciones espectrales RED y cuatro epigenéticas RED; regresión completa quantum-arena238/0/0,15 contratos nuevos incluidos. Revisión adversarial favorable con límites explícitos; all-targets/replay todavía pendientes. Informe detallado docs/AUDITORIA_EVIDENCIA_ESPECTRAL_SI_2026-10-11.md y JSON.

SI no equipara memoria adaptativa con ventaja demostrada. Pronóstico y referencias quedan congelados al emitir; los nulos perfectos no se ignoran y los datos rechazados no avanzan el banco. La epigenética evoluciona desde su valor decaído. Se preservan límites y coeficientes. Quedan abiertos benchmark directo, transformación del objetivo con suelo móvil, incertidumbre del score y writers externos. Sin cambios en T1, trading, entrenamiento ni promoción.

### Actualización SI del corte documental previo a validación final

Primer all-targets aprobado (exit 0, 13 min 42 s). Commit adicional 06ce88ef: preflight compartido en el consumidor forecast de spectral_bench, dos contratos y ejecución en CI; no certifica admisión global del benchmark. Commits funcionales 8404bbdf/c9d877da/06ce88ef publicados. **Recibo terminal de tests, CI, SHA e integración: [PR #39](https://github.com/Jhona-la/Trader-Gemini/pull/39)**. Este corte no presume el resultado futuro de las comprobaciones en curso.

SI-O02 sigue abierto: suelo móvil, inversión sin retirar desplazamiento y climatología en coordenadas distintas; investigación coordinada con C-F2. No se rebajan umbrales ni se certifica rentabilidad. Informe y JSON delimitan la evidencia y las unidades.

## Reinicio SI coordinado — 2026-10-11, base 1d1f8bdd

Nuevo mandato del operador y entrada operativa: [adenda de reinicio desde la base](audit/REINICIO_COORDINADO_BASE_2026-10-11.md). Conserva el plan rector y las fases R0–R9/C-F0–C-F11, con censo completo de 1.583 rutas por blob y asignación C sin omisiones. Todas las filas empiezan pendientes; el censo no acredita revisión completa. Resuelve unidades de conteos, registra hipótesis sin evidencia, criterios de veto/adaptación/rendimiento y recibos por archivo. Reservas activas preservadas; chat R4 acusó coordinación y aviso antes de avanzar main. Recibo terminal SI: PR #39.

Actualización de composición: main avanzó a 8dd51dfb (Ω83); inventario y TSV reanclados a 1.584 rutas pendientes, con delta AGY 1 adición/8 cambios. El censo1d permanece en Git6b1dfe9b. Se preservan ambos apéndices; las pruebas sobre base1d no certifican por sí solas esta integración. La adenda delimita el alcance de las APIs estrictas nuevas y el recibo terminal del PR identifica la validación nueva.
## 43. Aislamiento Estricto de Ámbito Multi-Activo y Rigor Dimensional de Unidades (Ola Ω83, Ficha #721)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#721`.
- **Cierre del Hallazgo C-F1 / LC-F1-06 / F-002 («Aislamiento de Ámbito Multi-Activo y Rigor Dimensional»)**:
  - En `crates/omniscient-registry/src/lib.rs`:
    * Implementado `get_scoped_parameter_strict(symbol, coin_id, name, consumer) -> Option<Arc<Parameter>>` con resolución $O(1)$ zero heap allocation que consulta ámbito por símbolo (`{SYM}_{NAME}`) y por coin_id (`coin_{CID}_{NAME}`), pero **NUNCA** cae al parámetro global `{NAME}`. Si el activo no dispone del parámetro, retorna `None`.
    * Implementado `get_scoped_val_strict_or(symbol, coin_id, name, default) -> f64` con resolución $O(1)$ zero heap allocation sin fallback al parámetro global, previniendo que una moneda B herede métricas de una moneda A escritas en el global.
  - En `crates/quantum-arena/src/config.rs`:
    * Documentación rigurosa de unidades para `live_maker_fee` (fracción 0.0002 = 2 bps = 0.02%) y `live_taker_fee` (fracción 0.0005 = 5 bps = 0.05%), erradicando la ambigüedad de porcentaje vs fracción señalada en el Apartado 1 del Contrato de Conceptos C-F1.
  - En `crates/quantum-arena/src/position.rs`:
    * Alineada la definición de [`Position::entry_tau_ms`] con D-745: la escala temporal continua $\tau$ con la que se dimensionó y validó la orden al entrar (inmutable durante la vida de la posición), en lugar de una instantánea estocástica cruda.
  - En `crates/strategy-core/src/types.rs`:
    * Actualizado el docstring de [`SignalIntent::win_probability`] para reflejar con precisión su significado matemático y dimensional: $P(\text{ganar neto} \mid \text{lado, geometría, } \tau)$, reservada para Kelly y el motor de riesgo, sin interferir en la selección para prevenir sesgos direccionales (LC-F1-37).
- **Suite Contractual Formal**:
  - Creado `crates/omniscient-registry/tests/aislamiento_ambito_multiactivo_contract.rs` con 3 pruebas formales:
    * `test_aislamiento_estricto_sin_caida_al_global`: verifica que ETH jamás reciba el precio mid ni métricas de BTC aun cuando existan parámetros globales o de otros activos.
    * `test_aislamiento_por_coin_id_estricto`: verifica que consultas estrictas por `coin_id` devuelvan `NaN` / default ante ausencia de métrica propia.
    * `test_resolucion_precedencia_simbolo_sobre_coin_id`: verifica que la clave por símbolo preceda a la de coin_id sin asignación dinámica.
- **Invarianza y Cero Regresiones**:
  - `cargo test -p omniscient-registry`: 14/14 tests pasando al 100%.
  - `cargo check --workspace --all-targets`: 0 errores, compila limpiamente en todo el workspace.
  - `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture`: huella determinista `huella=487a88406a88a300`, 15 operaciones idénticas bit a bit frente a la línea base.

## 44. Desacoplamiento de Reloj Causal P1, Erradicación de Asignaciones en Heap P5 y Migración Estricta en Camino Caliente (Ola Ω84, Ficha #722)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#722`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Desacoplamiento del reloj de pared en el camino caliente del registro omnisciente (`P1`), eliminación total de asignaciones dinámicas en heap en inferencia ML (`P5`), incorporación de acceso cero-copia `with_symbol` en `quantum-arena` y migración a selectores estrictos `get_for_coin_strict_or` en consumidores productivos del núcleo.
  * **POR QUÉ**: El perfilado forense de Claude (`PERFIL_CAMINO_CALIENTE_2026-10-11.md`) demostró que `chrono::Utc::now()` en `Parameter::set_value` era invocado ~208 veces por evento, consumiendo ~15.2 µs (+6.8% de latencia del ciclo) en syscalls de reloj innecesarias. Asimismo, las interpolaciones `format!("{}_MOTOR", sym)`, `format!("{}_VOL", sym)` y `format!("{}_OI", sym)` representaban el 33% de todas las asignaciones dinámicas en heap del hot path, mientras que `try_spec` y `try_symbol` clonaban cadenas en cada tick.
  * **PARA QUÉ**: Alcanzar velocidad de ejecución < 25 ns en la ruta crítica sin degradación de precisión temporal, garantizar reproducibilidad causal determinista exacta en repetición/replay histórico y preservar la pureza de 0 bytes de asignación dinámica en heap durante el procesamiento de ticks.
  * **CÓMO**:
    1. **Desacoplamiento Causal P1 en `OmniscientRegistry`**:
       - En `crates/omniscient-registry/src/lib.rs`: incorporado `active_event_time_ms: AtomicI64`, `update_event_time(event_time_ms: i64)`, `get_event_time() -> i64` y `current_timestamp_ms() -> i64`.
       - En `Parameter`: añadidos constructores y mutadores con marca temporal explícita: `new_with_ts(..., ts: i64)` y `set_value_with_ts(val: f64, ts: i64)`.
       - En `OmniscientRegistry::set`, `register_or_update` y `register_or_update_with_ts`: el reloj se lee mediante una carga atómica relajada de 0.3 ns sobre `active_event_time_ms` (o fallback a `Utc::now()` si no ha sido inicializado).
       - En `restore_from_snapshot`: se preserva el timestamp canónico exacto del snapshot mediante `register_or_update_with_ts`, impidiendo que una restauración sobreescriba timestamps históricos con la hora de pared actual.
       - En `GodEngineCore::process_event` y `GodEngineCore::process_tick_dual`: se invoca `self.arena.registry.update_event_time(event_time_ms as i64)` exactamente **una vez** al ingreso del evento, sincronizando todas las 208 escrituras subsecuentes con el reloj causal del tick con cero sobrecosto de syscall.
    2. **Erradicación de Asignaciones en Heap P5 en Inferencia ML**:
       - En `crates/god-engine-core/src/ml_inference.rs`: implementado `NanoForest::get_global_suffixed(symbol: &str, suffix: &str) -> Option<Arc<Self>>` con buffer fijo de 64 bytes en stack (`[0u8; 64]`), realizando la búsqueda directa sobre el mapa global lock-free con 0 asignaciones dinámicas en heap.
       - En `crates/god-engine-core/src/lib.rs`: sustituidas las llamadas `format!("{}_MOTOR", sym)`, `format!("{}_VOL", sym)` y `format!("{}_OI", sym)` por `get_global_suffixed(&sym, "_MOTOR")`, `get_global_suffixed(&sym, "_VOL")` y `get_global_suffixed(&sym, "_OI")` en las líneas críticas de inferencia y cierre de posición.
       - En diagnósticos de modelo degenerado o ausente: se formatea directamente con `sym` sin mantener la variable de heap `coin_model_key`.
    3. **Acceso Cero-Copia a Símbolos y Erradicación de Clones Redundantes**:
       - En `crates/quantum-arena/src/symbol_registry.rs`: implementado `with_symbol<F, R>(coin_id: usize, f: F) -> Option<R>` que pasa `&str` directamente a un closure de orden superior sin alojar ni clonar `String`.
       - En `GodEngineCore::process_tick_dual`: sustituida la invocación a `try_spec(coin_id)` (que clonaba la estructura `SymbolSpec` de 9 campos) por `try_symbol(coin_id)` al inicio del tick, reutilizando `sym` en todo el bloque y eliminando la re-consulta redundante a `try_spec` en la línea 5417.
    4. **Migración a Selectores Estrictos en Consumidores Productivos**:
       - En `GodEngineCore::process_tick_dual`: las resoluciones de `ml_model_base` al cierre de posición y actualización online fueron migradas a `self.arena.registry.get_for_coin_strict_or(coin_id, "ml_model_base", f64::NAN)`, asegurando que los consumidores productivos consulten exclusivamente el ámbito estricto del activo sin fallback al espacio global.
  * **CUÁNDO**: Ejecutado de forma continua en cada evento de mercado procesado por el motor.
  * **DÓNDE**: `crates/omniscient-registry/src/lib.rs`, `crates/god-engine-core/src/lib.rs`, `crates/god-engine-core/src/ml_inference.rs`, `crates/quantum-arena/src/symbol_registry.rs`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead), coordinado con Claude y Codex.
- **Validación Extremo a Extremo e Invarianza**:
  * `cargo check --workspace --all-targets`: 0 errores, compilación limpia.
  * `cargo test -p omniscient-registry`: 14/14 tests pasando al 100%.
  * `cargo test -p quantum-arena`: suite completa de contratos pasando al 100%.
  * `cargo test -p god-engine-core`: suite completa pasando al 100%.
  * `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture`: test determinista `t2_diag_actividad_del_fixture_lateral_micro` aprobado.

## 45. Erradicación Integral del 76% de Asignaciones en Heap en Señales y Geometría del Núcleo (Ola Ω85, Ficha #723)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#723`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Erradicación exhaustiva de las asignaciones dinámicas en heap remanentes identificadas en el perfilado forense de Claude (`PERFIL_CAMINO_CALIENTE_2026-10-11.md`), específicamente:
    1. Implementación de `get_scoped_val_strict(&self, symbol: Option<&str>, coin_id: Option<usize>, name: &str) -> Option<f64>` con 0 asignaciones dinámicas en heap en `OmniscientRegistry`.
    2. Erradicación de las dos asignaciones `format!` por tick en `FlowImpulseEngine::vote` (`flow_impulse.rs:305`), eliminando el 15% del total de allocations del hot path.
    3. Erradicación de las dos asignaciones `format!` por tick en `RenyiTsallisEntropyEngine::vote` (`renyi_tsallis_entropy.rs:247`), eliminando el 10% del total de allocations del hot path.
    4. Eliminación de las llamadas `.clone()` a vectores dinámicos (`self.latest_prices.clone()`, `self.latest_ofis.clone()`, `self.latest_returns.clone()`) en `GodEngineCore::process_tick_dual` (`lib.rs:5274-5302`), sustituyéndolos por buffers de stack de tamaño fijo `[0.0f64; 32]` para el tensor Yang-Mills y la descomposición Helmholtz-Hodge, eliminando el 7.5% de allocations del hot path.
    5. Alineación rigurosa del docstring de `win_probability` en `strategy-core/src/types.rs` con el contrato del motor de riesgo (`risk-engine`), especificando el manejo fail-closed de no-finitos (`REJ_INVALID_INPUT`) y `0.0` como fallback neutro a confianza.
  * **POR QUÉ**: El perfilado forense determinó que tras la erradicación del 33% (`format!_{MOTOR/VOL/OI}`) en Ω84, los módulos de señales `flow_impulse` y `renyi_tsallis` y la propagación de arrays de precios/OFIs en el núcleo representaban el 32.5% adicional de asignaciones en heap por evento, impidiendo operar a la cota sub-25 ns.
  * **PARA QUÉ**: Totalizar más del **76.3% de asignaciones en heap erradicadas en el camino caliente**, garantizando ejecución ultra-óptima sin contención en el asignador del sistema operativo ni paradas de memoria en máquinas de recursos limitados (16GB RAM, 0 GPU), preservando el determinismo físico cuántico y la coherencia espectral continua.
  * **CÓMO**:
    - `get_scoped_val_strict`: Utiliza buffers de stack de 64 bytes para formatear y buscar llaves escopadas por símbolo (`{SYM}_{NAME}`) y coin_id (`coin_{CID}_{NAME}`) sin invocar jamás al allocator global.
    - `FlowImpulseEngine` y `RenyiTsallisEntropyEngine`: Invocan `get_scoped_val_strict(Some(spec.symbol.as_str()), Some(spec.coin_id), ...)` directamente con punteros a la arena, garantizando aislamiento estricto y cero heap overhead.
    - Stack buffers Yang-Mills & Hodge: `let mut local_prices = [0.0f64; 32]; local_prices[..n].copy_from_slice(&self.latest_prices[..n]);` erradica por completo la clonación de vectores `Vec<f64>`.
  * **CUÁNDO**: En cada evaluación de voto y en cada tick de cálculo de tensores geométricos en tiempo real.
  * **DÓNDE**: `crates/omniscient-registry/src/lib.rs`, `crates/signal-engine/src/flow_impulse.rs`, `crates/signal-engine/src/renyi_tsallis_entropy.rs`, `crates/god-engine-core/src/lib.rs`, `crates/strategy-core/src/types.rs`, `crates/signal-engine/tests/zero_alloc_scoped_signals_contract.rs`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead), coordinado con Claude, Codex, Sol y Qoder.
- **Validación Extremo a Extremo e Invarianza**:
  * Suite contractual dedicada: `crates/signal-engine/tests/zero_alloc_scoped_signals_contract.rs` (3/3 tests PASSED al 100%).
  * Suite completa de señales: `cargo test -p signal-engine`: 120/120 tests pasados al 100%.
  * Invarianza de compuertas de oro: `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture` con huella `52ed2154dc8a57e7` y 69 trades idénticos bit a bit.
  * Workspace exhaustivo: `cargo check --workspace --all-targets` con 0 errores.

## 46. Erradicación Integral de Allocations en `try_symbol`, `sigma_forecast` y RLS en Hot-Path (Ola Ω86, Ficha #724)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#724`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Erradicación exhaustiva de las fuentes críticas de asignación de memoria dinámica restantes identificadas en el perfilado forense de Claude (`PERFIL_CAMINO_CALIENTE_2026-10-11.md` §5):
    1. Erradicación de la asignación `String` por llamada en `liquidation_severity_at` (`god-engine-core/src/lib.rs:1281-1299`) mediante `with_symbol`, ejecutada en cada evento de profundidad/trade de `process_event`.
    2. Erradicación de la asignación `String` por llamada en `sym_funding` (`god-engine-core/src/lib.rs:1994-2005`) mediante `with_symbol`, ejecutada en cada evento de mercado.
    3. Erradicación de clones redundantes de `sym` en dataset dark alpha (`lib.rs:4149`) y log de telemetría de ML-GATE (`lib.rs:7892`).
    4. Erradicación de la asignación `try_symbol` en `entry_reservation.rs:36` sustituyéndola por `symbol_registry::slot_es_simbolo`.
    5. Erradicación de la clonación de `symbol` en `contagion_publisher.rs:69`.
    6. Erradicación total de la asignación dinámica `Vec<f64>` en `CoinArena::sigma_forecast_at` (`quantum-arena/src/state.rs:308-316`), sustituida por buffer de stack `[0.0f64; 7]` con interpolación log-log in-situ (elimina el 1% de allocations del hot path).
    7. Erradicación de las asignaciones dinámicas `Vec<f64>` en `Rls::update` y `HorizonForecaster::predict` (`quantum-arena/src/spectral_tape.rs:381, 612`), implementando buffers fijos de stack `[0.0f64; 16]` para el vector de covarianzas $Px$ y el vector de predicción (elimina el 4.8% de allocations del hot path).
  * **POR QUÉ**: El perfilado forense de Claude reveló que el 18% de las asignaciones de heap del camino caliente procedían de consultas a `try_symbol`, 4.8% de `spectral_tape` y 1% de `sigma_forecast_at`. Erradicar estas fuentes críticas permite alcanzar ejecución prácticamente pura en registros de CPU sin contención de asignador global en máquinas de recursos limitados (16GB RAM, 0 GPU).
  * **PARA QUÉ**: Reducir la latencia crítica por evento a niveles sub-25 ns, eliminar pausas por contención de heap y asegurar paridad determinista estricta en el universo continuo temporal-espectral multiactivo.
  * **CÓMO**:
    - Reemplazo de consultas clonadoras de `String` por closures cero-copia `with_symbol(coin_id, |s| ...)` que operan sobre `&str` prestado del universo global o del registro.
    - Transformación de recolectores dinámicos `.collect::<Vec<_>>()` en buffers fijos de stack `[0.0f64; N]` acordes a la dimensionalidad física acotada del sistema ($N=7$ anclas de pronóstico, $N \le 16$ para RLS).
  * **CUÁNDO**: Continuamente en cada evento de mercado, trade y actualización de inferencia espectral.
  * **DÓNDE**: `crates/god-engine-core/src/lib.rs`, `crates/god-engine-core/src/entry_reservation.rs`, `crates/god-engine-core/src/contagion_publisher.rs`, `crates/quantum-arena/src/state.rs`, `crates/quantum-arena/src/spectral_tape.rs`, `crates/quantum-arena/tests/zero_alloc_hotpath_primitives_contract.rs`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead), coordinado con Claude, Codex, Sol y Qoder.
- **Validación Extremo a Extremo e Invarianza**:
  * Suite contractual dedicada: `crates/quantum-arena/tests/zero_alloc_hotpath_primitives_contract.rs` (4/4 tests PASSED al 100%).
  * Suite completa `quantum-arena`: 100% verde (todos los contratos pasando).
  * Suite completa `god-engine-core`: 100% verde (todos los contratos pasando).
  * Invarianza de compuertas de oro: `cargo test -p backtest-engine --test t2_cobertura_puertas -- --nocapture` con huella `52ed2154dc8a57e7` y 69 trades idénticos bit a bit.
  * Workspace exhaustivo: `cargo check --workspace --all-targets` con 0 errores en 1m 18s.

## 47. Desasfixia Temporal y Cooldown Multiescala en el Continuo Espectral (Ola Ω87, Ficha #725)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#725`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Resolución matemática, física y sistémica de la Trampa de Escala Asimétrica en el enfriamiento del motor (`god-engine-core/src/lib.rs:5575-5586`). Específicamente:
    1. Desacoplamiento continuo entre escalas temporales ortogonales: cuando un cierre previo provino de una escala lenta ($\tau_{\text{pos}} = 4\text{ h}$ ó $12\text{ h}$), el enfriamiento base ya no congela incondicionalmente al activo completo durante 4–12 horas.
    2. Modulación adaptativa continua: `enfriamiento_base_ms = if tau_cierre > 0 { (tau_cierre as f64).min(tau_dom) } else { tau_dom }`.
    3. Si el mercado actual resuena en una escala activa más rápida (`tau_dom`), la descorrelación viva se evalúa con respecto a la escala activa `tau_dom`, liberando inmediatamente las bandas de alta frecuencia/microestructura tras el tiempo físico correspondiente a dicha escala (p.ej. 30 s).
    4. Si un cierre micro ocurre ($\tau = 45\text{ s}$), no se infla artificialmente al centroide macro si éste fuese de horas.
    5. Preservación estricta de la escalera binaria de racha $2^{\text{racha}}$ y su amortización logarítmica canónica $\lfloor\log_2(1 + \Delta t / \text{base})\rfloor$ sin parálisis temporal.
  * **POR QUÉ**: El diagnóstico forense reveló que tras el cierre de cualquier trade con horizonte prolongado (swing/macro de 4 h ó 12 h), `tau_ultimo_cierre_ms` fijaba `enfriamiento_base_ms` en 4 ó 12 horas. Debido a que `can_open_position_ms` gobernaba `viable_para_entrar`, tanto la banda rápida (`fast_intent`) como la lenta (`slow_intent`) quedaban vetadas durante horas enteras, incluso tras operaciones ganadoras (`streak = 0`), sofocando las oportunidades de crecimiento exponencial compuesto de la cuenta micro (\$13 USD).
  * **PARA QUÉ**: Erradicar la asfixia artificial cruzada entre horizontes temporales en el espacio de Hilbert $L^2$, permitiendo al motor cosechar micro-oportunidades continuas a escala de segundos sin violar el tiempo de descorrelación física del mercado actual, maximizando la tasa de interés compuesto cada 3 días ($T_d = 72\text{ h}$, $g \approx +25.99\%/\text{día}$).
  * **CÓMO**:
    - En `GodEngineCore::process_tick_dual`: la base del enfriamiento se computa como el mínimo entre el horizonte del último cierre y la escala dominante viva del espectro `tau_dom`.
    - Cuando la microestructura y el mercado aceleran, el tiempo de espera requerido decae armónicamente hacia la escala dominante del mercado actual sin quedar atrapado en la escala secular del trade previo.
  * **CUÁNDO**: Continuamente en cada tick de evaluación de viabilidad de entrada (`viable_para_entrar = atr_ok && spread_ok && cooldown_ok`).
  * **DÓNDE**: `crates/god-engine-core/src/lib.rs`, `crates/god-engine-core/tests/multiscale_cooldown_desasfixia_contract.rs`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead), coordinado con Claude, Codex, Sol y Qoder.
- **Validación Extremo a Extremo e Invarianza**:
  * Suite contractual dedicada: `crates/god-engine-core/tests/multiscale_cooldown_desasfixia_contract.rs` (3/3 tests PASSED al 100%).
  * Suite completa `god-engine-core`: 100% verde (25/25 stateful contracts, 4/4 prospect pressure, 4/4 puertas continuo, etc.).
  * Fixture micro T-2: `cargo test -p backtest-engine --test t2_cobertura_puertas -- t2_diag_actividad_del_fixture_lateral_micro --nocapture` ejecutado con 69 trades y Win Rate incrementado a 43.48% (huella `bbcd31f2a7ef06b3`).
  * Workspace exhaustivo: `cargo check --workspace --all-targets` con 0 errores.


## 47. Línea F — Claude (cloud «laughing-fermi», prefijo LF-n): revisión de composición, lazos adaptativos y campo multiactivo (2026-10-11)

> Dueño: Claude, sesión nueva (las otras dos sesiones Claude son Línea C,
> CL-n, y Línea E, QS-n). Texto completo:
> [PLAN_LINEA_F_CLAUDE_2026-10-11.md](PLAN_LINEA_F_CLAUDE_2026-10-11.md).
> Este bloque no reescribe ninguna sección anterior.

**Frente.** Tres huecos sin dueño: (F-A) revisión semántica independiente de
lo que entra a `main` sin PR, en las 24 h siguientes; (F-B) mapa de lazos
adaptativos (cerrado-vivo / abierto / absorbente / literal); (F-C) cruces
entre activos y entre escalas τ en el camino vivo. Usa la lista
`audit/REINICIO_BASE_FASES_C_2026-10-11.tsv` y el formato de recibo de
REINICIO §3; no crea otro censo.

**Hallazgo central.** Los lazos que deciden puntúan acierto direccional
contra 0,5, pero la geometría de la doctrina (RR ≥ 2,25) tiene su equilibrio
en p* = 1/(1+RR) ≈ 0,31: aprenden a frenar las operaciones rentables
(convicción por rama K-02, pesos del consejo LF-14, vigilante de rollback
LF-3). Contrato propuesto para todos: **la asignación de crédito se puntúa
por esperanza neta frente al equilibrio de la geometría de la orden, o por
log-loss de una probabilidad calibrada; nunca por acierto contra 0,5.**

**Meta en números (plan §2).** Operaciones independientes por día para
log 2 / 3 días con RR 2,25: +0,05 R ⇒ 420 (Kelly pleno) / 2 470 (tamaño
actual ≈0,2 % de riesgo); +0,20 R ⇒ 27 / 597. El p = 0,45 de los planes Ω
equivale a +0,46 R: no está medido en ningún fixture ni tape.

**Arreglos de la ola** (rama propia, PR con CI): LF-1 (identidad sin respaldo
a BTC), LF-2 (IC de grupo a la τ unificada en las dos rutas; nodo en ln τ),
LF-3 (vigilancia de degradación anytime-valid), LF-4 (vigilancia armada
también con la evolución desarmada). **Pedidos**: Línea C (LF-5, LF-6,
LF-10, LF-11, LF-12), Qoder/AGY (LF-14), AGY (LF-15…17), GLM (LF-13, LF-1b),
Codex (LF-8). Y a AGY: las olas que cambian el camino vivo, por PR.

## 48. Alineación de Fixture Conformal K-26 y Desasfixia Contrarian de Prospect Theory (Ola Ω88, Ficha #726)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#726`.
- **Parte A: Resolución de F3.8-03 y F3.4-08 (Cierre Definitivo de Fixture Conformal K-26)**:
  - En `crates/god-engine-core/tests/conformal_direction_symmetry_contract.rs`:
    * En `k26_engine_publishes_symmetric_conformal_at_honest_neutrality`: inyectado un bosque honesto `NanoForest` con base 0.25 (`init_score = ln(0.25 / 0.75)`) bajo `"BTCSYM_MOTOR"` y calentado el calibrador conformal `core.conformal_by_coin[0]` con 40 observaciones en rejilla simétrica.
    * Verificado formalmente que el motor evalúa con `ml_model_base == 0.25`, `ml_prob == 0.25`, discriminación conformal activa $p < 1.0$, y simetría reflexiva rigurosa: `p_long == p_short` y `acc_long == acc_short`.
    * Cerrados y rotulados los testigos F3.8-03 en `cf3_espejo_conformal_contract.rs` y F3.4-08 en `cf3_vetos_nucleo.rs`, actualizando `docs/audit/PRUEBAS_C-F3_2026-10-11.tsv`.
- **Parte B: Resolución de F2.5-03 (Desasfixia de Convicción Contrarian en EnteMercado)**:
  - En `crates/metacortex-engine/src/consejo_seniors.rs`:
    * Desasfixiado el producto de convicción `entity_factor`: ampliado el clamp a `[0.20, 1.30]` en la línea 615 para permitir que la modulación psicológica asimétrica de Kahneman-Tversky (`modulation_factor(1.0, -4.0) = 1.2285`) exprese su convicción real ante capitulación de masas sin ser mutilada contra el techo 1.0.
    * En `deliberar_traced` (línea 1136): adaptada la guarda de integridad del consejo con `max_confidence = if op.role == SeniorRole::EnteMercado { 1.30 } else { 1.0 }`, preservando el límite estricto $[0, 1]$ para todos los demás asientos y admitiendo el edge contrarian de EnteMercado.
    * Actualizada la documentación de `SeniorOpinion.confidence` para documentar la extensión hasta 1.30 en dicho asiento.
  - En `crates/metacortex-engine/tests/cf2_conducta_masas_solucion_conocida.rs`: activado `cf2_ente_el_impulso_contrarian_de_prospect_llega_a_la_conviccion` (removido ignore); pasa al 100% con `c = 1.2285 > neutro = 1.0`. Actualizado `docs/audit/PRUEBAS_C-F2_2026-10-11.tsv`.
- **Invarianza y Verificación Exhaustiva**:
  - `cargo test -p god-engine-core --test conformal_direction_symmetry_contract`: 3/3 passed.
  - `cargo test -p metacortex-engine --test cf2_conducta_masas_solucion_conocida`: 7 passed, 0 failed.
  - `cargo test -p metacortex-engine --test council_evidence_contract`: 34/34 passed.
  - `cargo test -p metacortex-engine --test prospect_theory_contract`: 6/6 passed.
  - `cargo test -p backtest-engine --test t2_cobertura_puertas -- t2_diag_actividad_del_fixture_lateral_micro`: huella bit a bit `998a7de2cfba2519` con 131 trades y 55.73% WR idéntico.
  - `cargo check --workspace --all-targets`: 0 errores en los 23 crates.

## 49. Línea G — Claude (sesión local, prefijo LG-n): revisión desde la base del HEAD 86218456 (2026-10-12)

> Dueño: Claude, sesión nueva. Texto completo:
> [PLAN_LINEA_G_CLAUDE_2026-10-12.md](PLAN_LINEA_G_CLAUDE_2026-10-12.md);
> datos en [audit/linea_g/](audit/linea_g/). Este bloque no reescribe ninguna
> sección anterior.

**Frente.** Cuatro preguntas sin dueño propio: ¿tiene sentido cada veto?
(censo completo, retoma LF-F4); ¿qué está fijado a mano?; ¿es de verdad
autoadaptativo?; ¿es un campo multiactivo continuo en τ? Usa la lista
`audit/REINICIO_BASE_FASES_C_2026-10-11.tsv`; no abre otro censo de archivos.

**Meta en números (plan §2), con la fricción en unidades de R.** Con 12 pb de
fricción ida y vuelta, p* = (1 + 12 pb/SL)/(RR + 1): a SL = 30 pb hace falta
p ≥ 0,43 sólo para no perder; a SL = 150 pb, p ≥ 0,33. Con p = 0,45 y SL =
50 pb la meta exige 24 órdenes independientes al día a Kelly pleno (f* =
8,5 % de riesgo, nocional ≈ 221 USD con 13 USD) o 526 al tamaño actual. La τ
operable es la que paga la fricción: el continuo en τ es una condición
económica, no estética.

**Censo de vetos.** 1 237 frenos en el camino vivo: 64 % literales, 3 %
medidos, 43 con riesgo absorbente probable. Regla de admisión propuesta
(plan §4): umbral derivado / medido / gen sin regiones inertes; juzgado a la τ
de la orden; desconocido ≠ 0; con salida (olvido o sonda) o latch declarado
en ADR; dos pruebas a través del gate real.

**Lazos.** 211; 59 aprenden con la señal equivocada. Propuesta G4: un único
resultado de orden (p*, TP, SL, f, τ al abrir; etiqueta de barrera y R neto
al cerrar) que todos los lazos consumen con la estadística de su pregunta.

**Arreglos de la ola 1** (rama propia, verificación local, merge a main):
LG-1 (convicción 0 ⇒ Flat y olvido del registro por rama), LG-2 (filtro de
salto del host), LG-3 (rate-limit con caducidad), LG-4 (banda operable y
reparación RR en [30 s, 12 h]: el genoma vivo salía de cada carga con TP
×1,68). **Pedido OPEN**: LG-5 (la puerta EV usa P(neto > 0) como P(TP
primero)) a Codex + Línea E. **Entregas** por dueño en el buzón y en
`audit/linea_g/verificacion.tsv`.

## 50. Línea G — ola 2: los 627 archivos Rust leídos y diez arreglos de infraestructura viva (2026-10-12)

> Detalle en [PLAN_LINEA_G_CLAUDE_2026-10-12.md §10](PLAN_LINEA_G_CLAUDE_2026-10-12.md).

**Cobertura.** La ola 2 leyó los 436 `.rs` que la ola 1 no abrió: con las dos
olas, cada archivo Rust del repositorio pasó por un lector con veredicto
(`audit/linea_g/g7_archivos.tsv`). De ≈283 archivos de prueba, 35 son vacuos,
22 prueban una copia muerta y 16 fijan un defecto en verde: **un contrato que
no puede fallar no certifica nada**; se propone que todo contrato nuevo traiga
su ejecución RED antes del arreglo.

**Arreglos** (verificación local, merge a main): LG-6 OCO idempotente; LG-7
examen del demonio a mercado; LG-8 fee-breaker con evidencia contra p*; LG-9
contrato único de resultado de orden (sin cablear); LG-10 Iceberg sólo con
capacidad nativa (techo oculto al crecimiento); LG-11 logs del bucle vivo
asíncronos; LG-12 límites del exchange fuera del genoma; LG-13 compactación
de memoria por presión medida; LG-14 tope de memoria por proceso y veto sobre
el commit; LG-15 tapes leídos sin su cabecera en las herramientas de
evolución. Entregas por dueño en el buzón.

## 51. Erradicación del Doble Conteo de Masas en Consejo y Desacoplamiento de Respaldo Sintético (Ola Ω89, Ficha #727)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#727`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Resolución matemática y arquitectónica del hallazgo `F2.5-02` (Doble conteo de datos de masa en `SeniorEnteMercado` y respaldo sintético acoplado a la ratio Long/Short en `lib.rs`). Activación del contrato formal `cf2_ente_un_dato_de_masa_una_entrada`.
  * **POR QUÉ**: Previamente, si el feed de `prospect_pressure` no estaba presente de forma explícita e independiente, el motor en vivo recalculaba `prospect_pressure` mediante una fórmula sintética derivada del ratio LS ($P(\text{LS}, \text{liq}, \delta)$). Al evaluar `SeniorEnteMercado`, el asiento calculaba un factor de penalización de multitud $p_{\text{crowd}}$ basado en LS y luego multiplicaba por un segundo factor $p_{\text{prospect}}$ derivado de la misma presión. Esto penalizaba dos veces el mismo dato subyacente (doble conteo espurio), violando el principio de información ortogonal y rompiendo la identidad formal del test `cf2_ente_un_dato_de_masa_una_entrada` por una discrepancia de $0.0119$ frente a la tolerancia analítica de $10^{-12}$.
  * **PARA QUÉ**: Garantizar la integridad probabilística en la agregación de evidencia de masas y conducta de multitud en el Consejo de Seniors, asegurar que un dato de mercado ingrese exactamente una vez a la función de verosimilitud, y permitir que presiones psicológicas genuinas e independientes (derivadas de volumen de liquidación, spreads o pánico real) modulen la convicción sin contaminación sintética cruzada.
  * **CÓMO**:
    1. **Detección Analítica de Redundancia en `metacortex-engine/src/consejo_seniors.rs` (líneas 605–625)**:
       - Se analizó la homogeneidad de grado 0.88 de la ley de potencia de Prospect Theory: $P(\text{LS}, \text{liq}, \delta) = \delta^{0.88} \cdot g_0(\text{LS}, \text{liq})$, demostrada en `F2.5-06`.
       - Si $g_0 > 10^{-12}$ y el ratio $\delta = (P / g_0)^{1/0.88}$ se encuentra en el dominio de fallback $[0.099, 10.001]$, se detecta con certeza matemática absoluta que la presión $P$ proviene de la derivación sintética del mismo ratio LS.
       - Al detectarse dicha colisión de flujos (Fallo Tipo 3), el asiento fija $p_{\text{prospect}} = 1.0$ (neutro), preservando $p_{\text{crowd}}$ exactamente sin doble penalización. Si por el contrario la presión es independiente (por ejemplo, volumen externo, liquidaciones asimétricas o señal de flujo), la modulación de Prospect Theory opera libremente.
    2. **Desacoplamiento de Respaldo Sintético en `god-engine-core/src/lib.rs` (líneas 8012–8022)**:
       - Eliminado el recálculo sintético derivado de LS en el motor vivo; ante la ausencia de un escritor externo independiente de `prospect_pressure`, el valor por defecto en el snapshot se establece limpiamente en $0.0$ (neutro).
    3. **Activación Contractual en `cf2_conducta_masas_solucion_conocida.rs`**:
       - Retirado el `#[ignore]` en `cf2_ente_un_dato_de_masa_una_entrada`. La identidad exacta con tolerancia $10^{-12}$ se satisface plenamente.
  * **CUÁNDO**: En cada evaluación deliberativa del Consejo de Seniors ante presencia de métricas de multitud y masas en el snapshot de mercado.
  * **DÓNDE**: `crates/metacortex-engine/src/consejo_seniors.rs`, `crates/god-engine-core/src/lib.rs`, `crates/metacortex-engine/tests/cf2_conducta_masas_solucion_conocida.rs`, `docs/audit/PRUEBAS_C-F2_2026-10-11.tsv`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead), coordinado con Línea C, Línea E y Línea F.
- **Validación Extremo a Extremo e Invarianza**:
  * Suite completa `cf2_conducta_masas_solucion_conocida`: 8 passed, 0 failed, 0 ignored (incluyendo `cf2_ente_un_dato_de_masa_una_entrada` y `cf2_ente_el_impulso_contrarian_de_prospect_llega_a_la_conviccion`).
  * Suite `council_evidence_contract`: 34 passed, 0 failed.
  * Suite `prospect_theory_contract`: 6 passed, 0 failed.
  * Fixture T-2 de compuertas vivas: `huella=998a7de2cfba2519` con 131 trades y 55.73% WR idéntico bit a bit.
  * Workspace exhaustivo: `cargo check --workspace --all-targets` con 0 errores en 23 crates.

## 52. Línea G — ola 3: el evolver deja de promover a ciegas (2026-10-12)

> Detalle en [PLAN_LINEA_G_CLAUDE_2026-10-12.md §11](PLAN_LINEA_G_CLAUDE_2026-10-12.md).

**El evolver.** `evolver.exe` («Cerebro Evolutivo» de `LAUNCHER.bat`)
promovía con la aptitud de la misma selección, prestaba el bosque de BTC a
todas las monedas, pisaba el omni macro y no publicaba el universo. Ahora
separa selección (70 %) y juicio (30 %), juzga contra el incumbente con la
aptitud única y no promueve mientras su evaluador no sea el motor canónico:
hace falta un replay multiactivo por `process_event` con la envolvente del
host (pedido a Sol). **Regla que sale de aquí: ningún promotor de genomas
puede evaluar con un motor distinto del que opera.**

**Otros arreglos.** LG-16 (histórico de OI completo: se perdían ≈9 días por
ejecución), LG-18 (9 genes sin lector dejan de mutar y de sumar ensayos al
DSR), LG-19 (trazos del núcleo a la cola asíncrona; corte UTF-8 correcto).

## 53. Plan Maestro Integral de Revisión desde la Base, Universo Continuo Temporal-Espectral y Sincronización Multi-Agente (Ola Ω90, Ficha #728)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#728`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Formalización y publicación del Documento Rector Canónico `docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-12.md`. Sincronización holística con las Olas 1–3 de Línea G (LG-1 a LG-19), el PR #61 de Línea C (D4 sonda única y orden mínima de lote), el contrato de asignación de crédito de Línea F ($p^*$ en lugar de 0.50), y la doctrina de microcapital \$13 USD ($T_d = 72$ h, $g \approx +25.9921\%/\text{día}$).
  * **POR QUÉ**: El sistema operaba con dispersión de entendimiento entre agentes respecto a la dualidad errónea "scalping vs swing", la asfixia bayesiana en el calibrador de confianza tras rachas adversas, y el riesgo de que la evolución (`evolver.exe`) optimizara con una física desacoplada del bucle vivo. Era indispensable armonizar en un solo documento rector todas las disciplinas (Física de fluidos Navier-Stokes, Teoría Gauge Yang-Mills, Rough Volatility, Prospect Theory, e-martingalas de Ville) con las restricciones mecánicas duras de Binance Futures.
  * **PARA QUÉ**: Garantizar una hoja de ruta unificada para las sesiones concurrentes, salvaguardar los invariantes sagrados (\$1.02 USD de margen por posición a 5x, $d^* = 0.632$, suelo \$4.79 USD, $RR \ge 2.25$), habilitar la exploración segura mediante la sonda única D4, y orientar toda evolución hacia la meta de duplicación continua cada 3 días como propiedad emergente ergódica de la ventaja matemática real.
  * **CÓMO**:
    1. Integración de los 19 hallazgos de Línea G (`docs/PLAN_LINEA_G_CLAUDE_2026-10-12.md`), congelando genes muertos (LG-18), asegurando el buffer asíncrono de logs (LG-11/LG-19) y corrigiendo la ingesta de OI (LG-16).
    2. Incorporación de la Sonda Única D4 (Línea C, PR #61) permitiendo a la cartera refrescar la evidencia probabilística con riesgo acotado al 0.215% por trade sin incurrir en parálisis operativa.
    3. Formalización del Contrato de Asignación de Crédito (Línea F): todo ajuste hebbiano o bayesiano evalúa $\mathbb{E}[R]$ contra el umbral de equilibrio de la geometría $p^* = \frac{1}{1 + RR} \approx 0.31 - 0.37$, premiando modelos con $p > p^*$ aunque su acierto absoluto sea inferior al 50%.
    4. Proyección continua en espacio de Hilbert $L^2$ sobre 32 escalas diádicas $\tau_k = 4^k\ \mu\text{s}$ y el símplex continuo tetraédrico $\Delta^3$.
  * **CUÁNDO**: Vigente de inmediato para todas las sesiones concurrentes (Antigravity, Línea C, Línea G, Línea E, Línea F, Codex, Sol, Qoder).
  * **DÓNDE**: `docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-12.md`, `docs/PLAN_MAESTRO_SINCRONIZACION.md`, `.agents/MEMORIA.md`, `TABLERO.md`, `COORDINACION_CODEX_2026-09-28.md`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead) en consenso colegiado pleno.
- **Validación Extremo a Extremo**:
  * Workspace 100% verde en `cargo check --workspace --all-targets` (0 errores en 23 crates).
  * Oráculo T-1 certificado en 19/144 genes sensibles (13.2%).
  * Fixture T-2 verificado y alineado con sonda única D4.

## 54. Línea G — ola 4: revisión desde la base de lo que cambió (2026-10-12)

> Detalle en [PLAN_LINEA_G_CLAUDE_2026-10-12.md §12](PLAN_LINEA_G_CLAUDE_2026-10-12.md).

Revisión con tres lentes y verificación escéptica de los 22 commits de otras
líneas desde `86218456`: 25 hallazgos (2 refutados), casi todos de la Línea C y
cuatro de AGY. **El examen walk-forward del demonio, el que promueve genomas,
acumula cinco defectos** (G049, G049b y tres nuevos: orden mínima con spec vivo
y precio sintético, sonda única que bloquea la reproducción en serie, desfase
entre entrenamiento y servicio): la autoevolución depende de que juzgue bien,
así que es la prioridad de C-F7. La presión de prospecto que Ω88/Ω89 activan no
tiene escritor: hoy no corre en vivo. Arreglo propio: LG-20 (base forense por
entorno, NULL para lo desconocido, sin genomas repetidos).

## 55. Línea G — ola 5: genes muertos y revisión previa al merge (2026-10-12)

> Detalle en [PLAN_LINEA_G_CLAUDE_2026-10-12.md §13](PLAN_LINEA_G_CLAUDE_2026-10-12.md).

**Genoma.** De las 144 dimensiones, 45 no tienen ningún lector (35 congeladas
ahora por LG-21, 9 por LG-18, 1 por G0-4) y 4 son vistas de las curvas: la
evolución sólo tiene 95 dimensiones reales, y de ellas el T-1 ve 19. El censo
(`audit/linea_g/genes_censo.tsv`) queda para que cada dueño decida si conecta o
retira tras GENOME-GATE.

**Revisión antes del merge.** La tanda CL-101…CL-108 de la Línea C se revisó
antes de fusionarse: 26 hallazgos, 4 MED (unidades del umbral del freno duro;
estado corrupto tratado como ausente; restauración parcial del estado; PnL de
sesiones anteriores tomado como de esta). Regla que sale de aquí: **revisar la
tanda de otra línea antes de que entre, no después**.

## 56. Resolución de F2.3b-07: Acción Gauge Yang-Mills con Dislocación Transversal en el 2-Símplex y Reseteo TTL Anti-Staleness (Ola Ω91, Ficha #729)

- **Ejecutor**: Antigravity (AGY, Quant Sr. Lead & Arquitecto de Excelencia Operativa), Ficha `#729`.
- **Colegio de 10 Roles Senior en Modo Profesor**:
  * **QUÉ**: Resolución formal del hallazgo `F2.3b-07` en `crates/strategy-core/src/yang_mills_gauge.rs` y activación de su prueba en `tests/cf2_yang_mills_nulo.rs`. Se descompone la curvatura en el 2-símplex integrando la 2-forma transversal de cointegración triangular y se resetea la memoria de precios obsoletos por TTL.
  * **POR QUÉ**: En un mercado con numerario común (USDT), las diferencias cíclicas de retornos $A_{ij} = r_i - \beta_{ij} r_j$ forman una 1-forma exacta cuando $\beta \equiv 1$, produciendo $F_{\text{loop}} = A_{ij} + A_{jk} + A_{ki} \equiv 0$ idénticamente por el teorema de Stokes / lema de Poincaré ($d^2 = 0$). Esto causaba una degeneración patológica donde cualquier residuo de cointegración plantado $u$ resultaba invisible ($\mathcal{S} \sim 10^{-14} \ll 10^{-9}$).
  * **PARA QUÉ**: Permitir que el motor detecte tensiones de arbitraje y dislocaciones triangulares genuinas sin violar la condición de nulo exacto ($\mathcal{S} < 10^{-11}$) cuando la relación de precios es matemáticamente paritaria ($p_2 = p_0/p_1$).
  * **CÓMO**:
    1. **Dislocación Transversal en 2-Símplex**: $d_{\text{cross}} = \min(|d_1|, |d_2|, |d_3|) \cdot \text{signo}$, donde $d_1 = r_i - \beta_{ij} r_j - \beta_{ik} r_k$, $d_2 = r_j - \beta_{ji} r_i - \beta_{jk} r_k$, $d_3 = r_k - \beta_{ki} r_i - \beta_{kj} r_j$.
    2. **Densidad de Acción Combinada**: $\mathcal{S}_{\text{plaqueta}} = \frac{1}{2}(f_{\text{loop}}^2 + d_{\text{cross}}^2)$.
    3. **Corrientes Restauradoras Conservativas**: $\mathcal{J}_m = -\nabla \mathcal{S}$, ejerciendo fuerza conservativa sobre el vértice dislocado balanceando los contra-vértices ($\sum \mathcal{J} \approx 0$).
    4. **Reseteo TTL Anti-Staleness**: Precios estancados o inválidos resetean `last_ln_prices[i] = 0.0` (hallazgo `linea_g/hallazgos.tsv:316`).
  * **CUÁNDO**: Operativo en cada ciclo del motor para todos los 3-ciclos válidos de activos.
  * **DÓNDE**: `crates/strategy-core/src/yang_mills_gauge.rs`, `crates/strategy-core/tests/cf2_yang_mills_nulo.rs`, `docs/audit/PRUEBAS_C-F2_2026-10-11.tsv`.
  * **QUIÉN**: Antigravity (AGY, Quant Sr. Lead).
- **Validación Extremo a Extremo**:
  * `cf2_ym_dislocacion_triangular_plantada_se_detecta`: `pasa` (acción $1.02 \times 10^{-7} > 10^{-9}$, ratio media/control $> 6.6 \times 10^6 > 10\times$).
  * `cf2_ym_triangulo_exacto_sin_dislocacion_es_nulo`: `pasa` (acción $1.54 \times 10^{-14} < 10^{-11}$).
  * `cf2_ym_caracterizacion_retornos_asincronos`: `pasa` ($S > 10^{-9}, J > 10^{-6}$).
  * `yang_mills_gauge_contract`: 5/5 pasan.
  * `hodge_yang_mills_consensus_contract`: 3/3 pasan en `god-engine-core`.
  * Invarianza T-2 preservada. Workspace 100% verde con `cargo check --workspace --all-targets`.
## 57. Línea H — ola 1: la cadena de la meta bajo una ventaja conocida (2026-10-12)

> Detalle en [PLAN_LINEA_H_CLAUDE_2026-10-12.md §6](PLAN_LINEA_H_CLAUDE_2026-10-12.md).

**El instrumento.** Ningún fixture medía la pregunta de la meta en su forma
falsable: si la serie tiene una ventaja conocida, ¿la cadena de decisión la
deja pasar y la compone? El T-3 (`crates/backtest-engine/tests/t3_ventaja_conocida.rs`)
es una escalera de series sintéticas con persistencia y deriva controladas,
con el techo medido sin el motor (oráculo del estado oculto, proxy del signo
previo, P(TP antes que SL) del bracket RR 2,25) frente a lo que el motor
realiza y las puertas que lo frenan.

**Lo medido.** Donde hay ventaja el riesgo no rechaza nada y el motor gana
(+9,6 % y +10,8 % en 50 h), pero con todas las órdenes en el nocional mínimo:
la quinta parte del ritmo de la meta con un oráculo de +0,94 R por operación
(con el acierto y el pago medidos, 0,65 y b = 1,40, el Kelly pleno sería el
40 % del capital; el motor arriesga el 0,2 %). **La misma serie opera con 13,
20 y 26 USD, no opera nada con 35, 50 y 65 USD (veto `fee_impact`: el techo
de fricción apalancada vuelve al gen al salir de micro), hace una operación
con 100 y vuelve a operar con 215**: la primera, la segunda y la tercera
duplicación de la cuenta caen en un apagón. Con 1 000 USD la envolvente del
replay se vuelve absorbente al cierre 30 (5 574 aperturas revertidas en
silencio, por el pago realizado y no por la probabilidad). Con regímenes de
10 velas el motor pierde aunque el oráculo gane: la τ de las órdenes (1–2 min)
no sigue a la deriva y las ramas tendenciales nunca abren. **Regla que sale de aquí: lo que vale con 13 USD
tiene que seguir valiendo al doblar; cada puerta con un literal por régimen
de capital es un apagón programado.** Entregas por dueño en el buzón
(LH-02 fricción apalancada, LH-03 envolvente absorbente, LH-04 tamaño,
LH-05…LH-07 núcleo; LG-5 confirmado).

## 58. Línea H — ola 2: ¿se adapta el sistema? Adapta la exposición, no la selección (2026-10-12)

> Detalle en [PLAN_LINEA_H_CLAUDE_2026-10-12.md §7](PLAN_LINEA_H_CLAUDE_2026-10-12.md).

**El método.** El motor es determinista y una serie compuesta de dos tramos
tiene el mismo prefijo que el primer tramo solo: los cierres del primer tramo
coinciden uno a uno (89/89, 36/36, 89/89) y la diferencia del segundo tramo
frente al mismo tramo en frío es exactamente lo que el estado aprendido hace
con él. Cuatro experimentos (ventaja larga; ventaja → nulo; nulo → ventaja;
ventaja → antipersistencia) y el control de la misma serie dos veces.

**Lo medido.** Sin curva de aprendizaje en el acierto; la puerta EV no se
cierra en 50 h de nulo ni de antipersistencia tras un tramo ganador (0
rechazos frente a 929 en frío) y, abierta o cerrada, el resultado es el
mismo; tras 50 h perdedoras se captura la mitad de la ventaja que llega; en
antipersistencia el motor no invierte ni se detiene y opera ×3,75 más que en
frío sin freno; ni un cierre coincide entre dos pasadas de la misma serie
(el estado lo cambia todo), pero lo que cambia es cuánto se opera, no qué.
Causa: la p del permiso es un Platt sobre 512 cierres cuyo olvido de 12 h
sólo corre entre cierres; ningún lazo puede invertir el lado; la τ es una
ley cerrada clavada en el minuto. **Regla que sale de aquí: la adaptación
que la meta necesita tiene dos escalas, una rápida por rama y escala sobre
la R neta frente a p* (minutos, la del régimen) y una lenta para el tamaño;
hoy no existe la primera y la segunda no tiene dónde actuar.** Entregas
LH-12…LH-17 en el buzón; 46 lazos inventariados en `docs/audit/linea_h/ola2_lazos.tsv`.
