# PLAN MAESTRO DE SINCRONIZACIÓN — Crecimiento Exponencial Compuesto

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

## 0. La meta y su física

**100% cada 3 días desde 13 USD, por interés compuesto.** En log: 0.231/día.
Requiere Sharpe diario ≈ 0.68 a Kelly pleno sin fricción — los mejores
fondos operan en 2-6 anualizado. **La meta no se discute: se sirve.** Lo
que exige de nosotros es excelencia en cada capa: ningún bps regalado en
ejecución, ningún edge no medido, ninguna señal no espectral.

La estrategia honesta hacia la meta (consenso de seniors, 2026-09-28):
maximizar el crecimiento log con tope de ruina y **medirlo fuera de
muestra**. El T-1 mide expresividad genética, no rentabilidad; la
cadena de paridades bt↔vivo (GLM, 10/10) garantiza que lo medido es lo
que opera. El eslabón que falta es el edge OOS estable — ese es el foco
de las tres líneas de trabajo de abajo.

## 0b. Barrido exhaustivo (mandato 2026-10-04)

El operador ordenó recorrer TODOS los archivos uno por uno en fases
(metas → conceptos → matemática → física → código):
**docs/BARRIDO_EXHAUSTIVO_FASES.md** — 8 fases, ~377 archivos, ~137k
líneas, división por línea dueña, verificación por fase.

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

## 3. Cadencia hacia la meta

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
| 2026-10-04 | Qoder | PRE-FLIGHT: regresión del árbol combinado 7c4cea40 = 828/0 + ws 0 err; roster 18 MOTOR (BTCUSDT incluido); T-1 de sello sobre el tip exacto en vuelo. Riesgos aceptados documentados en buzón. |
| 2026-10-04 | GLM | LXXXIV (en vuelo, lxxxiv-l2v1): datasets τ-matched ago+sep-14 EN GENERACIÓN (nota de regeneración atendida) + trainer v1 logística con plantilla honesta; paridad del PR#27 de Claude anunciada. |
| 2026-10-04 | Codex | root-audit-2026-10-04 (en vuelo): 16 hallazgos root-contract (parser risk-state, límites OOS, tokens escalares, agregación depth); reconciliado con main 7c4cea40 (este plan) tras all-targets check; documenta gates de certificación sin resolver. PR #28 (G0-G8) aprobado por GLM en LXXXV. |
| 2026-10-04 | GLM | LXXXV cerrado: L2 v1 PARCIAL DEFINITIVO — dirección transfiere OOS (+5 pts constante) pero calibración NO (Platt sobrefía selección); cableado BLOQUEADO; fase 4 opcional por petición del consejo. Mapa de planes §5b publicado (tres documentos, roles distintos). |
| 2026-10-04 | Qoder | SELLO FINAL: T-1 del tip exacto 7c4cea40 PASA 16/144 (4221s) — pre-flight VERDE completo, sistema listo para operar (§4.2). Regresión 828/0 + roster 18 MOTOR. |
| 2026-10-04 | GLM | LXXXVIII: saga L2 cierra HERMÉTICA (condicionado≈global; señal concentrada en no-range +8.7 pts). DECISION_FDUSD_BORRADOR.md publicado — esperando palabra del dueño (recomendada B: archivo). |
| 2026-10-05 | Antigravity | PLAN MAESTRO QUANT SR ACTUALIZADO: formalización física y matemática del Universo Continuo Espectral $S(\omega, \tau, \mathbf{x}, t)$, Hodge, Fokker-Planck, E-Values de Ville, Lo-MacKinlay Variance Ratio y DSR Bailey & LdP. Barrido F0-F7 CERRADO (171 hallazgos acumulados), F8 (136 tests de integración) abierta. Micro-cuentas $13 USD auditadas. |
| 2026-10-05 | GLM | LXXXXVIII: CI rojo reparado tras F4-H2 — tests cl41b y live_envelope_gate actualizados a la conducta de bootstrap al apalancamiento validado. Limpieza de residuo de stash. |
| 2026-10-06 | Qoder | OLA 61 CERRADA (a01227cc): F2-B2/B3/B4 sustrato espectral honesto — habilidad_en vecino ln τ, W₁ a lag FÍSICO 60 s (anillo por timestamp del exchange, cadencia 250 ms), ζ(p) con peso continuo de masa. Oráculo PASA 16/144 (2212 s). Banda operable = nodos 18..22 de la malla 4^k µs. |
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
