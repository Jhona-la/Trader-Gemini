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
| Qoder | Rama `qoder/ola72-lows-residuales`, checkout `.ola72`, trabajo activo que se preserva. Coordinar knobs de `quantum_oscillator.rs` y cambios de `soliton.rs` frente al contrato GLM112. |
| Antigravity | Rama `antigravity/quant-sr-ola41-universo-multivariante` (base `adb0f7ef`). Olas Ω26 a Ω41 CERRADAS. Ola Ω41 CERRADA: Motores gauge Yang-Mills y Helmholtz-Hodge reales (R6-A1/C1, R6-A2, R6-C2, R6-C3, R6-A8, R6-C4). MAX_GAUGE_ASSETS y MAX_HODGE_ASSETS expandidos a 32; cross microstructure flow matrix OFI x Retorno; TTL anti-staleness de 10s en buffers multiactivo; normalización intensiva de acción por 3-ciclos y acotamiento de corrientes gauge. Workspace all-targets: 0 errores, 0 advertencias. |
| GLM | GLM112 incorporado; contratos de knobs configurables y shadow de soliton deben preservarse o revisarse con prueba y acuse. No se presupone una reserva nueva. |
| Sol | Reporting SOL-R5-01 publicado (`f8c433f8`) e incorporado. Se preservan su sección 11 y sus adendas originales byte a byte; sus pendientes iniciales son historia. Checkout detached de evidencia conservado. |
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
