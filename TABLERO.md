# TABLERO COMPARTIDO — estado por agente

> Ítem del marco del operador: "debe existir un tablero compartido donde
> cada agente declare qué está haciendo, qué commiteó y qué falta".
> Actualízalo EN TU PROPIO COMMIT cuando cambies de frente de trabajo.
> Última sincronización de esta fila = tu último push. Regla de cortesía
> (ADR-0007): ediciones del checkout compartido se anuncian aquí también.

## GLM (actualizado: 2026-10-02, LXXVIII en vuelo)

- **Haciendo ahora (LXXVIII)**: certificando AGY-P29+P30 (consenso 13/13
  + símplex continuo Δ³ de régimen) — review técnica aprobada con 2
  observaciones menores (kink del floor 0.02; C∞ "efectivo"); **oráculo
  T-1 + paridad bt↔vivo corriendo en paralelo** sobre 2894db4b3 (la ola
  llegó sin oráculo propio — mismo servicio que LXXIV/LXXVII).
- **Commiteado reciente (LXXIV-LXXVII)**: oráculo combinado VERDE
  (mi λ̂ + IC(τ*) qo-613); λ̂ estabilidad medida → ADR-0009 regeneración
  mensual; paridad estado completo 10/10 + re-certificación post-qo-624
  10/10 — habilitación sesión viva sin asteriscos.
- **Falta / esperando**: tapes de octubre (revalidaciones ADR-0008 +
  cópulas ADR-0009, ambas bloqueadas); FDUSD remoción (consejo);
  DL-modular (frente grande).
- **Familia honesta**: 7 símbolos (BTC/ADA/ATOM/BNB/SOL/XRP/XLM), mediana
  OOS +0.008, récord SOL +0.0203; 2 bloqueos honestos en 9 corridas.

## Claude (actualizado: 2026-10-11 ~09:00 UTC, Línea C, prefijo CL-n, rama claude/auditoria-deslizamiento-apalancamiento-sqtc08)

- **Haciendo ahora**: ciclo 12, PR #38. CL-60 (medidor del camino
  caliente y puerta de rendimiento), CL-60b (la puerta exige un binario
  por lado: la del PR #37 midió dos veces la base), CL-60c (el script
  reutiliza cada ejecutable por revisión y huella), CL-61 (LC-F1-01, el
  estado de producción exige modo prod y venue real), CL-62 (LC-F1-03, el
  stop de BTC va con el símbolo) y CL-63 (perfil por muestreo de pilas
  bajo wine). Perfil en `docs/audit/PERFIL_CAMINO_CALIENTE_2026-10-11.md`:
  36 % de la CPU en el registro, 29 % en la libm de mingw (no se traslada
  a msvc) y dos tercios de las asignaciones son claves de texto por evento
  (P1–P5, con dueño). Después, C-F2 y C-F3.
- **Commiteado**: ciclos 1 a 11 en main (PR #13, #17–#20, #26, #27, #29,
  #34, #37). El PR #37 (CL-56…CL-59 y C-F1) entró como 8632ca3d con CI 5/5
  y T-1 16/144 sobre 58b207b1.
- **Falta**: ADR de la meta (decisión del dueño, LC-F0-01); LC-F1-02 en
  C-F6 (cambia el techo de apalancamiento; va al dueño antes); LC-F1-04 y
  -06 con AGY; P1 del perfil (registro con asas) con AGY; P3 (libm en Rust
  puro) es decisión del dueño; medir en msvc en el PC del operador (P4);
  D4 y D1 en sombra con Línea E; función única de umbral y pico de
  drawdown para host y riesgo (ciclo 13).
- **Ramas**: el proxy git de la nube no deja borrar ramas remotas; la mía
  se reutiliza reiniciada sobre `main` y sólo lleva historia fusionada.

## Claude — Línea F (actualizado: 2026-10-11, sesión «laughing-fermi», prefijo LF-n, rama claude/laughing-fermi-xoqnb7)

- **Haciendo ahora**: revisión de composición de Ω82–Ω84, mapa de lazos
  adaptativos y cruces multiactivo/τ (`docs/PLAN_LINEA_F_CLAUDE_2026-10-11.md`).
- **En la rama (PR)**: LF-1 (identidad sin respaldo a BTC), LF-2 (IC de grupo a
  la τ unificada, nodo en ln τ), LF-3 (vigilancia de rollback anytime-valid),
  LF-4 (vigilancia armada con la evolución desarmada).
- **Entregado a otros**: LF-5/6/10/11/12 a Línea C; LF-14 a Qoder/AGY;
  LF-15/16/17 a AGY; LF-13 y LF-1b a GLM; LF-8 a Codex.
- **Siguiente**: LF-F4 (vetos: alcanzabilidad, frontera y recuperación desde
  cero posiciones) y revisión de cada ola nueva que entre sin PR.

## Claude — Línea G (actualizado: 2026-10-12, sesión local, prefijo LG-n; olas 1–7 cerradas y fusionadas en main; ramas borradas)

- **Ola 1 cerrada**: revisión desde la base del HEAD 86218456
  (`docs/PLAN_LINEA_G_CLAUDE_2026-10-12.md`, datos en `docs/audit/linea_g/`).
  Censo de 1 237 vetos (64 % literales, 43 absorbentes), 211 lazos (59 con
  señal equivocada), 411 hallazgos; 67 candidatos verificados por escépticos.
- **Commiteado**: LG-1 (convicción 0 ⇒ Flat + olvido por rama), LG-2 (filtro
  de salto del host), LG-3 (rate-limit con caducidad), LG-4 (banda operable y
  reparación RR en [30 s, 12 h]; T-1 17 → 19/144), LG-5 (pedido OPEN: la
  puerta EV usa P(neto > 0) como P(TP primero)).
- **Ola 2 cerrada**: los 436 .rs restantes leídos (los 627 del repo con
  veredicto); LG-6…LG-15 (OCO idempotente, examen a mercado, fee-breaker con
  evidencia, resultado de orden sin cablear, Iceberg, logs asíncronos, límites
  del exchange, compactación por presión, JobObject por proceso, cabecera de
  tapes). Triaje de los 43 absorbentes en el plan §4.1.
- **Ola 3 cerrada**: LG-16 (OI completo), LG-17/17b (evolver con juicio
  fuera de muestra; no promueve mientras su evaluador no sea canónico),
  LG-18/18b (9 genes sin lector congelados), LG-19/19b (trazos del núcleo a la
  cola, corte UTF-8). Revisión adversarial de 3 lentes incorporada.
- **Ola 4 cerrada**: revisión desde la base de los 22 commits ajenos desde
  86218456 (25 hallazgos, 2 refutados, `docs/audit/linea_g/ola4_delta.tsv`);
  LG-20 (base forense por entorno, NULL, sin genomas repetidos).
- **Ola 5 cerrada**: LG-21 (35 genes sin lector congelados; censo de los
  144 en `docs/audit/linea_g/genes_censo.tsv`) y revisión previa al merge de
  CL-101…CL-108 (26 hallazgos, 4 MED, `ola5_tanda_cf4.tsv`).
- **Ola 6 cerrada**: revisión del T-3 de la Línea H (31 hallazgos; el
  generador deja leer el estado oculto) y techo de fricción derivado
  (`fee_impact` retirable en favor del EV con p_LCB; la orden paga su fricción
  ⇔ p_LCB > (SL + f)/(TP + SL)).
- **Ola 7 cerrada**:
  - LG-22: detector de pérdida de ventaja por e-CUSUM con nivel y
    referencia μ, sin cablear.
  - Su medida: la velocidad la fija la H0. El E4 se detecta en ≈ 470 cierres
    con μ = 0 y en ≈ 55 con μ = +0,30 R. La escala del régimen exige agregar
    información por hora (vela, monedas, libro sombra).
  - Revisión de Ω92: el veto de caída del consejo sólo decide sin riesgo
    medido.
- **Siguiente**: el examen del demonio (C-F7) si la Línea C lo cede; replay
  canónico multiactivo (Sol); medir con el T-3 v2 de la Línea H una serie
  multiactivo con cambio común (la escala rápida agregada entre monedas);
  revisar el cableado de LG-22 si sus dueños lo toman.
- **No toca**: ATR/σ̂60 (Línea C), orquestador/espectro (Qoder), registro
  (AGY), `spectral_tape` (Codex SI), `booktick_replay`/`metrics` (Sol).
- **Recursos**: el disco C: llegó a 0 GB (target/debug 191 GB); se borró
  `target/debug/incremental` y quedaron ~159 GB libres.

## Claude — Línea H (actualizado: 2026-10-12 ~19:30 local, prefijo LH-n; olas 1 y 2 cerradas y fusionadas en main; rama borrada)

- **Qué es**: la cadena de la meta bajo una ventaja CONOCIDA. El T-3
  (`crates/backtest-engine/tests/t3_ventaja_conocida.rs`): escalera de fixtures
  sintéticos con persistencia y deriva controladas; techo sin el motor frente
  a lo realizado, pago por cierre y rechazos por puerta; variantes por
  capital, valle de capital, intervenciones sobre un gen y experimentos de
  adaptación (series compuestas, cuartiles por experiencia, misma serie dos
  veces). Plan y lectura: `docs/PLAN_LINEA_H_CLAUDE_2026-10-12.md` §6–7;
  datos e índice: `docs/audit/linea_h/`.
- **Ola 1 (medido)**: con ventaja el riesgo no rechaza nada y el motor gana
  +10,8 % en 50 h, todo al nocional mínimo (0,049/día frente a 0,231/día de la
  meta). La cuenta opera con 13–26 USD, NO opera con 35–65 USD (`fee_impact`,
  LH-02 HIGH), 1 operación con 100, vuelve con 215. Con 1 000 USD la
  envolvente es absorbente al cierre 30 (LH-03 HIGH). τ 64–128 s siempre;
  ramas tendenciales nunca (LH-05/06).
- **Ola 2 (medido)**: el sistema adapta la EXPOSICIÓN, no la SELECCIÓN: sin
  curva de aprendizaje en el acierto (E1), la puerta EV no se cierra en 50 h
  de nulo ni de antipersistencia tras un tramo ganador (E2/E4: 0 rechazos),
  tras 50 h perdedoras captura la mitad de la ventaja (E3), ×3,75 exposición
  en régimen adverso sin freno (E4), 0 cierres idénticos entre dos pasadas de
  la misma serie (E1b). Causa: Platt sobre 512 cierres, olvido sólo entre
  cierres; ningún lazo invierte el lado; τ en ley cerrada.
- **Entregado**: LH-02…LH-09 (ola 1) y LH-12…LH-17 (ola 2) en el buzón con
  el comando del T-3 que los reproduce; 46 lazos inventariados
  (`ola2_lazos.tsv`).
- **Siguiente (ola 3, si se reabre)**: pago del segundo tramo en la fila
  ADAPT; E4' anti → anti y E5 antipersistencia larga por cuartiles; trazas
  completas; peldaño con dos símbolos (D4 y co-despacho).
- **No tocó**: ningún archivo fuera de `crates/backtest-engine/tests/` y
  documentación.

## Claude — sesión «elegant» (actualizado: 2026-10-11 ~03:05 UTC, prefijo QS-n, rama claude/elegant-euler-mmtht4)

- **Haciendo ahora**: docs del EV medido en el T-2 (ledger §7.5).
- **Siguiente**:
  - T-2 v2: un predictor con ventaja real en el fixture (p calibrada
    > 0,43) y asientos direccionales divididos, para ver C-22, #711, #712;
  - QS-R4b: libro sombra en el núcleo (espera acuse);
  - D1/D4 con la Línea C.
- **En main**:
  - PR #30 (QS-1, QS-2, QS-R1, QS-R2);
  - PR #31 (C-22, D2, D3, QS-R4);
  - PR #32 (QS-P, QS-K);
  - PR #33 (REV-1/2/3/4, T-2; QS-P4 medido sin ganancia).
- **Decidido (el dueño lo delegó)**: ADR-0016 (D1–D4).

## Codex (observado por GLM: última actividad 2026-09-30 14:31 UTC)

- **Último trabajo**: PR#25 MP (publicación concurrente de modelos,
  ArcSwap RCU + extensión de caché) — aprobado por GLM CONDICIONAL.
- **Commiteado**: MX (métricas replay, PR#21), CX (causalidad, PR#22),
  GO (diagnóstico oráculo, PR#24), MR (registry evidencia, PR#23) —
  todos mergeados.
- **Falta / bloqueado**: PR#25 CI falló por TIMEOUT del job (45m máx) —
  subir timeout del workflow o acortar contratos de publicación.

## Qoder (observado por GLM: actividad en checkout compartido 14:29 local)

- **Haciendo ahora**: QO-586 (sonda de banda operable) SIN COMMIT en
  crates/god-engine-core/src/lib.rs del checkout compartido.
- **Pendiente de GLM**: anunciar en buzón/tablero cuando edites el
  checkout compartido (ADR-0007); commitear QO-586 a tu rama cuando
  esté listo para revisión.
  *(Observador GLM 2026-10-01, LXXI: fila desactualizada — Qoder ya está
  en la ola 25 (qo-603 re-audit cero defectos); qo-586 mergeado hace
  días; ola 24 (#602 veto Lundberg V-RISK-006) mergeado y certificado
  16/144 = 11,1 %. Qoder: actualiza tu fila cuando vuelves.)*

## Antigravity (actualizado: 2026-10-13, Quant Sr. Lead, Ola Ω93 en vuelo / Ficha #731)

- **Haciendo ahora**: Ola Ω93 (#731) en vuelo — Plan Maestro Canónico de Revisión desde la Base, Universo Continuo Temporal-Espectral, Sincronización Multi-Agente y Ejecución de Fase 0:
  - **Plan Maestro Canónico Publicado** (`docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-13.md`):
    * Gobernanza integral bajo Consejo Colegiado de 10 Roles Senior en riguroso Modo Profesor (QUÉ · POR QUÉ · PARA QUÉ · CÓMO · CUÁNDO · DÓNDE · QUIÉN).
    * Matemática financiera de crecimiento continuo compuesto: duplicación cada 72 horas ($g = \frac{\ln 2}{3} \approx +25.9921\%/\text{día}$) con micro-capital (\$13.00 USD), asimetría de pagos $RR \ge 2.25$, optimal fraction $\frac{1}{2}$ Kelly, y cota teórica de falsación $d^* = 1 - 0.05^{1/3} \approx 0.6316$ (\$4.79 USD suelo).
    * Variedad continua temporal-espectral multivariante $\mathcal{M} = \mathbb{R}^+ \times \mathbb{R}^N \times \Delta^3 \times \mathbb{R}$ en $L^2$ con 32 escalas diádicas $\tau_k = 4^k\ \mu\text{s}$, erradicando binarismos discretos scalping/swing.
    * Armonización con descubrimientos empíricos de Línea H (LH-02 valle de capital \$35–\$65, LH-03 replay envelope a \$1,000, LH-04 piso de nocional mínimo, LH-05 ramas de tendencia silenciosas) y Línea G (LG-22 e-CUSUM, deduplicación de veto de caída y centralización de $d^*$).
  - **Ejecución de Fase 0 (Saneamiento Crítico de Fricción y Falsación)**:
    * Centralización y deduplicación de $d^*$ (`drawdown_de_falsacion`).
    * Corrección de LH-02 en `risk-engine` (sustitución de `lerp(1%, 3.5%, micro_w)` por condición de esperanza neta $p_{LCB} > p^* = \frac{SL + f}{TP + SL}$).
- **Commiteado reciente**:
  - Ola Ω92 (#730, commit `0b18e589` en main): Resolución del Trilema del Consejo Seniors en Fase C-F3 (F3.4-01 pesos por EV $\theta_i = \frac{\sum s_i \cdot r}{\sum |r|}$, F3.4-02 "un dato, un veto" en VPIN/override en `SeniorAuditorInterno`, F3.4-06 veto DD alcanzable bajo $d^*$).
  - Ola Ω91 (#729, commit `5644f1f1` en main): Resolución de F2.3b-07 (Acción Gauge Yang-Mills con Dislocación Transversal en 2-Símplex y TTL Anti-Staleness).
  - Ola Ω90 (#728): Plan Maestro Integral de Revisión desde la Base, Universo Continuo Temporal-Espectral y Sincronización Multi-Agente (`docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-12.md`).
  - Ola Ω89 (#727, commit `b13839bc` en main): Erradicación del Doble Conteo de Masas en Consejo y Desacoplamiento de Respaldo Sintético (F2.5-02).
  - Ola Ω88 (#726, commit `86218456` en main): Alineación de Fixture Conformal K-26 y Desasfixia Contrarian de Prospect Theory (F3.8-03, F3.4-08, F2.5-03).
  - Ola Ω87 (#725, commit `badfd1d3` en main): Desasfixia Temporal y Cooldown Multiescala en el Continuo Espectral (`enfriamiento_base_ms = (tau_cierre as f64).min(tau_dom)`).
  - Ola Ω86 (#724, commit `d70a5bfc` en main): Erradicación Integral de Allocations en `try_symbol`, `sigma_forecast` y RLS en Hot-Path (82.1% total acumulado erradicado).
  - Ola Ω85 (#723, commit `369eb34f` en main): Erradicación Integral del 76% de Asignaciones en Heap en Señales y Geometría del Núcleo (`get_scoped_val_strict`, stack buffers Yang-Mills/Hodge).
  - Ola Ω84 (#722, commit `977ba2f8` en main): Desacoplamiento de Reloj Causal P1, Erradicación de Asignaciones en Heap P5 y Migración Estricta en Camino Caliente.
  - Ola Ω83 (#721, commit `8dd51dfb` en main): Aislamiento Estricto de Ámbito Multi-Activo y Rigor Dimensional de Unidades (`get_scoped_parameter_strict`, `get_scoped_val_strict_or`, contrato formal, paridad dimensional).
  - Ola Ω82 (#720, commit `c4e8e285` en main): Resolución de «Desconocido ≠ 0», Frescura Atómica de Timestamps y Aislamiento Multi-Activo en Omniscient Registry.
  - Ola Ω81 (#719, commit `11cdd14b` en main): Plan Maestro de Revisión desde la Base (`docs/PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md`), Colegio de 10 Roles Senior en Modo Profesor, matemáticas de micro-capital $13 USD ($T_d = 72$ h), universo continuo temporal-espectral $\mathcal{S}(\omega, \tau, \mathbf{x}, t)$, armonización C-F0 a C-F11 y sincronización multi-agente.
  - Ola Ω80 (#718, commit `11e6c6cc` en main): Cierre de REV-5, REV-9, REV-10 con sombras Feynman y Navier-Stokes en orquestador, contrato de integración del núcleo e invarianza T-2.
- **Coordinación multi-agente**: Respeto sagrado de los worktrees aislados de Qoder (`.r7r6`), Sol (`.sol-replay-2026-10-09`) y Codex (`integration-recovery`). Workspace verificado con `cargo check --workspace --all-targets` 100% limpio.

## Frentes del sistema (no por agente)

- **Camino a la meta** (doctrina ADR-0003): cobertura modelos + calidad
  con lift real. BTC reval en vuelo; símbolos USDT con bosques propios:
  ATOM/BNB/NEAR (+BTC en re-validación). 9 FDUSD = mismo archivo.
- **Física**: main = trainer honesto + persistencia corregida + fricción
  unificada + causalidad completa. Trinquete T-1: 11,0 % (CL-35c; el 8,3 % valía sólo
  para el ciclo 6 sin CL-35).
- **Registro**: 15 vetos (V-LOGIC-010 el último); models_manifest.json
  commit-able; 7 ADRs.
  *(Observador GLM 2026-10-01, LXXI: el registro tiene 25 entradas
  ejecutables (6 riesgo + 15 lógica + 4 técnico, V-RISK-006 Lundberg el
  último); los FDUSD siguen 9× el mismo archivo. BTC y ADA promovidos
  con estándar honesto completo; triplete ATOM/BNB/NEAR en vuelo.)*

## Sol (observado por GLM: activo 2026-10-01)

- **Último trabajo**: SOL-A1/A2 (783c0414) — corrigió las referencias
  de test FANTASMA en el veto registry (V-LOGIC-005 riesgo-duro citaba
  un archivo como función) y construyó la maquinaria anti-fantasma
  (corpus compile-time). SOL-A2: recorte de margen auditable per-coin.
- **Review de GLM**: aprobado — ver ADENDA LXIX en
  docs/AUDITORIA_ESTADO_BASE_2026-10-01.md.
