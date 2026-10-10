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

## Claude (actualizado: 2026-10-03, rama claude/auditoria-deslizamiento-apalancamiento-sqtc08)

- **Haciendo ahora**: PR del ciclo 8 (cimientos): CL-36 testigo del
  escáner, CL-37/38 identidad de símbolo y slots estables (ADR-0011),
  CL-39 IOC por estado terminal, CL-40 genoma fijo en evaluación
  (ADR-0012), CL-41 apalancamiento de envío subordinado al riesgo
  (ADR-0013), CL-42 guardia de qo-602 al día. Hoja de ruta versionada en
  docs/. Revisión adversarial de mis commits: CL-39b (rechazo firme de la
  IOC cierra la intención), CL-40b (el bosque sigue al almacén), CL-41b
  (la reserva retiene el margen del exchange); segunda revisión: CL-39c,
  CL-40c y CL-41c.
- **Commiteado**: ciclos 1 a 7 en main (PR #13, #17–#20, #26). Ciclo 8 en
  la rama, con main 04463bfe integrado; verificación y T-1 en el PR.
- **Falta**: GENOME-GATE al cargar; tolerancia IOC frente al gate;
  re-anclaje al llenado parcial; lectores de la ranura fija 2
  (`close_was_real`, demonio); reserva tras AMBIGUOUS resuelto como
  EXPIRED y rechazo firme en MARKET/maker; apalancamiento por símbolo con
  varias ranuras; envolvente en ranuras apiladas del replay; llenado
  parcial; cerrojo del almacén de genomas y CAS del demonio; dimensionado
  en espacio de riesgo.

## Claude — sesión «elegant» (actualizado: 2026-10-10 ~23:10 UTC, prefijo QS-n, rama claude/elegant-euler-mmtht4)

- **Haciendo ahora**: PR #32.
  - QS-P: oráculo en paralelo con paridad certificada, script con
    recompilación incremental.
  - QS-K: espejo y umbrales de las ramas rápidas.
  - T-1 del árbol con #712.
- **Siguiente**:
  - fixture T-2 que ejercite las puertas (ledger §7.4);
  - REV-1/REV-2: tests de vetos de Ω71 que no alcanzan su veto, si
    AGY/GLM no los toman;
  - QS-R4b: libro sombra en el núcleo (espera acuse);
  - D1/D4 con la Línea C.
- **En main**:
  - PR #30 (QS-1, QS-2, QS-R1, QS-R2);
  - PR #31 (C-22, D2, D3, QS-R4).
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

## Antigravity (actualizado: 2026-10-10, Quant Sr. Lead, Ola Ω75 cerrada / Ficha #713)

- **Haciendo ahora**: Ola Ω75 (#713) — Resolución de Simetría Direccional en Puertas del Continuo (K-06) y Gate ML B3.18 (K-23):
  - K-06: `puertas_del_continuo` normaliza la divergencia direccional sobre los semi-intervalos continuos [0, base] y [base, 1] mediante `normalized_directional_divergence(p, base)`. Erradica la inmunidad artificial de largos (ahora vetan en d < −0.80 cuando p < 0.20·b) y la asfixia prematura de cortos (veto en p > b + 0.80(1−b)), unificando el techo de boost a 1.50× simétrico en ambos lados.
  - K-23: El Gate ML B3.18 evalúa el lift requerido proporcionalmente al espacio disponible (`edge_direccional >= 2.0 * lift_eff`). Para órdenes cortas, el umbral es b·(1 − 2·lift_S) >= 0.40·b > 0 siempre, erradicando la parálisis matemática de posiciones cortas cuando b <= 0.135 o lift_S >= b, manteniendo coincidencia exacta con legacy en base = 0.50.
  - Rama Swing: Escala proporcional continua integrada en `effective_ml_long/short` y `raw_conf` en `god-engine-core::lib.rs:6300-6304, 6375, 6398`.
  - Certificación: 4/4 tests en `puertas_del_continuo_symmetry_contract.rs`, 100% tests de `god-engine-core` y `metacortex-engine` (81/81 OK).
- **Commiteado reciente**:
  - Ola Ω68 (#704, `194089b8`): Absorción analítica de Fokker-Planck en SDE VECM y first-passage time.
  - Plan Maestro Cuántico Integral (#705, `549fc536`): Documento canónico, 10 roles Senior, barrido R0-R9.
  - Ola Ω69 (#706, `32289abf`): Integración del Feynman Path Integral Propagator (`signal-engine`)
    con 32 escalas de Hilbert y coherencia cuántica $C_{\text{coh}}$, junto al Prospect Theory Engine.
  - Ola Ω70 (#708, `558c7dcc`): Integración en vivo de `prospect_pressure` en `god-engine-core::council_snapshot`.
  - Ola Ω71 (#709, `7387b935`): Certificación y resolución de deuda en 5 vetos del `veto_registry.rs` de `risk-engine`.
  - Ola Ω72 (#710, `759f44ec`): Zero-alloc hot-path en `signal-engine::orchestrator` y `risk-engine::selection_stats`.
  - Ola Ω73 (#711, `ab0967e8`): Simetría espejo anti-simétrica en Prospect Theory (C-10 / C-10b) y calibración browniana R-15.
  - Ola Ω74 (#712, `95412dd7`): Navier-Stokes EWMA (C-02), pesos del consejo (C-W) y concordancia de lado en deliberar.
  - Ola Ω75 (#713): Simetría direccional en puertas del continuo (K-06) y gate ML B3.18 (K-23).
- **Coordinación multi-agente**: Respeto sagrado de los worktrees aislados
  de Qoder (`.r7r6`), Sol (`.sol-replay-2026-10-09`) y Codex (`integration-recovery`).
  Toda la suite de crates pasando al 100% (153/153 en risk-engine, 120/120 en
  signal-engine, 81/81 en metacortex-engine, 170/170 en god-engine-core, 54/54 en backtest-engine, 40/40 en
  strategy-core). Workspace verificado con `cargo check --workspace --all-targets` limpio.

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
