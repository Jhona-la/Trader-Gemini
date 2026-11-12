# Revisión V36 desde la base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir el estado real del repositorio en una secuencia verificable de correcciones causales que termine en evidencia Demo/OOS neta, sin relajar gates ni perder trabajo multiagente.

**Architecture:** Trabajo incremental sobre overlays e índices privados, con un sujeto inmutable por recibo y fusión `git merge-tree` + CAS. V2 permanece como guarda; V3r2 sólo diagnostica. Las teorías nuevas quedan subordinadas a propiedad, causalidad, consumidor vivo y superioridad OOS.

**Tech Stack:** Rust workspace estable con nightly `2026-06-30`, Cargo/rust-lld, Python de contratos existentes, Git/GitHub CLI y `.agent_tx`.

**Spec:** `docs/superpowers/specs/2026-11-05-base-review-v36-design.md`

## Global Constraints

- Objetivo de investigación +100 %/72 h sujeto a evidencia OOS después de fees, funding y slippage; nunca garantía ni instrucción de riesgo.
- `main` y `origin/main` se revalidan antes de cada CAS; ningún PASS se transfiere entre SHA, WIP, perfil o toolchain distintos.
- V2 conserva fixture, comparador, ABI 144 y mínimo `0.110`. V3r2 no sustituye el gate ni rehabilita R28.
- No editar QRA/Navier/`adversarial_gate`/`scripts/tests/test_pre_push_gate.py` ni WIP ajenos. No `git add -A`, reset/stash/checkout forzado, force-push, `--no-verify` o purgas.
- No Demo/Mainnet, Producción, lectura de credenciales, kills de procesos, borrado de overlays/worktrees o limpieza por tamaño/antigüedad. Ausencia = `N/D`.
- Todo cambio de producción exige RED previo, GREEN posterior y revisión independiente; `check` no equivale a prueba ni a attestation.

## Review Focus

- Proyección genética borrada/acoplada: conservar solicitud/realización y no otorgar sensibilidad por proxy (Task 1 y Task 4).
- Sujeto mixto: vincular SHA, WIP, perfil y comandos antes de aceptar evidencia (Tasks 2 y 5).
- Pérdida de trabajo: toda rama/worktree exige propietario, archivos, lease y rescate antes de liberar (Task 3).
- Ausencia económica fabricada: ROI/PnL/WR/Sharpe/EV inválidos se reportan `N/D`, no cero (Tasks 6 y 7).
- Métrica convertida en objetivo: ningún gen, fixture o teoría se modifica para aumentar el conteo sin una hipótesis financiera (Tasks 4 y 6).

---

### Task 1: Corte de verdad y censo V2

**Files:**
- Create: `.agent_tx/staging/KIMI-QUANT-SR-V35/V2_COORDINATE_CENSUS.json`
- Read: `.agent_tx/staging/KIMI-QUANT-SR-V35/r28-handoff-v2-full-workspace-tests-resumed.log`

**Interfaces:**
- Consumes: los registros `[T1-PROBE]` y `[T1-STATS]` del oráculo V2 ya ejecutado.
- Produces: 144 filas con `requested`, `realized`, `changed_slots`, `projection`, `request_adjustment` y `observed_hash_change`.

- [x] **Step 1: Verificar R28 contra su base**

Run: comparación de las 144 huellas antes/después de R28.
Expected: mismas 144 huellas y mismos 14 sensibles; R28 no es una regresión demostrada.

- [x] **Step 2: Clasificar las coordenadas sin alterar el gate**

Expected: 14 cambios de huella, 4 acopladas, 4 borradas y 5 ajustes; sin reabrir fixture ni umbral.

- [x] **Step 3: Verificar lectores, no inferir desconexión del conteo**

Run: Kelly `cl163_tope_kelly_lh04_contract`, curvas `f1_073_atomic_horizon_curves_seqlock_contract` y `horizon_reader_parity`.
Expected: 5/5, 3/3 y 6/6; medición auxiliar 15/15. Ninguna reconexión justificada por el número.

- [ ] **Step 4: Entregar el censo a QRA como evidencia, no como veredicto de contrato**

Expected: archivo JSON disponible y citado en el bus multiagente.

### Task 2: Publicar diseño, plan y estado por CAS documental

**Files:**
- Create: `docs/superpowers/specs/2026-11-05-base-review-v36-design.md`
- Create: `docs/superpowers/plans/2026-11-05-base-review-v36.md`
- Create: `.agent_tx/diagnostics/base-review-v36-status.json`

**Interfaces:**
- Consumes: Task 1 y refs frescas.
- Produces: plan canónico y manifiesto de sujeto para cualquier agente que retome el trabajo.

- [ ] **Step 1: Revisar el diseño y el plan con un builder independiente**

Expected: ningún Critical/Important sin resolver; cualquier supuesto queda explícito.

- [ ] **Step 2: Revalidar `HEAD`, `origin/main`, worktrees y hashes de los documentos**

Run: `git ls-remote --heads origin`, `git rev-parse HEAD`, `git worktree list --porcelain`.
Expected: el plan nombra el corte real o registra la deriva y se recompone.

- [ ] **Step 3: Escribir árbol con índice privado, fusionar con `git merge-tree` y revisar ambos padres**

Expected: sólo documentos propios; WIP ajeno e índice canónico intactos.

- [ ] **Step 4: Ejecutar CAS documental sin push**

Expected: `refs/heads/main` avanza al commit revisado con expected SHA. No attestation ni CI presumidos.

### Task 3: Ramas y recursos con procedencia antes de cualquier liberación

**Files:**
- Create: `.agent_tx/diagnostics/base-review-v36-branches.json`
- Create: `.agent_tx/diagnostics/base-review-v36-resources.json`

**Interfaces:**
- Consumes: `git worktree list --porcelain`, PR/ancestría, RAM y censo de disco disponible.
- Produces: bloqueos y candidatos condicionados; nunca una orden de borrado por antigüedad.

- [ ] **Step 1: Registrar cada rama y su bloqueo**

Expected: `linea-i-i13` se conserva por PR #171 abierto y 5 commits exclusivos; todas las demás conservan worktree hasta revisión de archivos y liberación.

- [ ] **Step 2: Registrar RAM y límites de medición de disco**

Expected: ≈2,17 GiB libres de 23,38 GiB; disco actual `N/D` por timeout de ~13 minutos; no se reutilizan tamaños históricos como actuales.

- [ ] **Step 3: Clasificar artefactos sin borrar**

Expected: `data/`, `models/`, `.git/`, recibos, logs activos, locks, índices, worktrees y trabajo no rescatado = intocables. Cachés fríos sólo candidatos tras propiedad/procedencia.

- [ ] **Step 4: Aplicar la regla de eliminación sólo cuando se cumplan todas las guardas**

Expected: tip ancestro de `origin/main` fresco, cero commits exclusivos, PR/merge comprobado, worktree libre, archivos rescatados y ausencia de lease/reserva. Si falta una condición, no se borra.

### Task 4: Decisión contractual V2 con QRA

**Files:**
- Read: `.agent_tx/staging/KIMI-QUANT-SR-V35/V2_COORDINATE_CENSUS.json`
- Create: `.agent_tx/diagnostics/t1-v2-qra-decision-request.json`

**Interfaces:**
- Consumes: censo V2 y revisión independiente V3r2.
- Produces: decisión explícita de contrato antes de cualquier cambio de producción.

- [ ] **Step 1: Entregar el censo y la prueba de que V3r2 no equivale al gate**

Expected: filas históricas C0=13, Cminus=15, Cplus=13 y unión=17 visibles; no se presenta la unión como recuperación V2.

- [ ] **Step 2: Exigir clasificación de cada coordenada antes de proponer corrección**

Expected: proyección borrada/acoplada, ajuste/clamp, ausencia de efecto o lector funcional; no “gen muerto” inferido sólo por hash.

- [ ] **Step 3: Elegir una única resolución documentada**

Expected: conservar V2 y deuda, o revisión formal que mantenga una guarda local equivalente y pruebe no-regresión bajo fixture original. QRA decide; este plan no la sustituye.

- [ ] **Step 4: Si existe desconexión demostrada, abrir RED específico; si no, no editar producción**

Expected: ningún cambio de genes, sizing, fees o fixture por Goodhart.

### Task 5: Mapa de attestation y primera corrección autorizada

**Files:**
- Read: `scripts/pre_push_gate.py`, `scripts/tests/test_pre_push_gate.py` (sin editar)
- Create: `.agent_tx/staging/KIMI-QUANT-SR-V35/ATTESTATION_GAP_MAP.json`
- Create: archivo de contrato elegido sólo tras una brecha demostrada y propietario libre.

**Interfaces:**
- Consumes: Tasks 1–4 y contratos de publicación existentes.
- Produces: fases faltantes por sujeto y una única corrección RED→GREEN con propietario libre.

- [ ] **Step 1: Ejecutar los contratos existentes sin modificar sus fuentes**

Run: `python -I -B scripts/tests/test_pre_push_gate.py -v`.
Expected: resultados completos por nombre, sin convertir fallos en PASS ni atribuir WIP ajeno al candidato.

- [ ] **Step 2: Registrar seis fases, Python legacy, Golden9, Clippy/formato y V2 por separado**

Expected: mapa de huecos con evidencia y propietario; attestation sigue falsa hasta cierre real.

- [ ] **Step 3: Seleccionar una brecha libre con hipótesis causal y escribir un test que falle**

Expected: contrato concreto con entrada/salida y archivo de producción identificado; sin selección por conveniencia.

- [ ] **Step 4: Implementar el cambio mínimo y repetir el contrato**

Expected: mismo test RED→GREEN, sin refactor colateral ni supresión de warnings.

- [ ] **Step 5: Revisar y hacer CAS sólo del lote validado**

Expected: formato, alcance de tests exigido, revisión independiente, diff contra cada padre y CAS con expected SHA.

### Task 6: Arranque, fees y evolución online con resultados explícitos

**Files:**
- Read: `crates/god-engine-core/src/bootloader.rs`
- Read: `crates/audit-engine/src/telemetry.rs`
- Read: `crates/evolution-engine/src/online_daemon.rs`
- Create: `.agent_tx/diagnostics/startup-fees-online-audit.json`

**Interfaces:**
- Consumes: baseline attestada y átomos libres de Tasks 4–5.
- Produces: mapa de ignición, fricción, ADN y aprendizaje online con evidencia y propietario por hallazgo.

- [ ] **Step 1: Comprobar orden de ignición y ausencia de fases ficticias**

Expected: integridad/ADN → balances/posiciones → universo → warmup → entrenamiento/modelos → operación; cada fase con evidencia real o fallo cerrado.

- [ ] **Step 2: Auditar ROI/PnL/WR antes/después de fees, EV/trade, Sharpe, periodo, dataset y commit**

Expected: valores medidos o `N/D`; ninguna ausencia convertida en cero ni clamp presentado como ley financiera.

- [ ] **Step 3: Comprobar persistencia/restauración completa y aprendizaje online en vivo**

Expected: modelos completos con versión coherente, cancelación/drenaje y ausencia de publicaciones parciales; backtest solo no acredita aprendizaje vivo.

- [ ] **Step 4: Abrir átomos RED→GREEN sólo para defectos demostrados**

Expected: cada cambio tiene consumidor vivo, baseline, coste y prueba de reinicio/paridad; sin teoría decorativa.

### Task 7: Shadow/Demo/OOS y cierre final

**Files:**
- Create: `CERTIFICATION_REPORT.json` sólo cuando exista evidencia real.
- Create: `.agent_tx/diagnostics/final-attestation.json`

**Interfaces:**
- Consumes: todas las correcciones integradas y una suite completa verde.
- Produces: attestation del SHA final, publicación y CI del mismo SHA.

- [ ] **Step 1: Ejecutar workspace completo y Zero-Wait sobre el candidato final**

Run: `cargo +nightly-2026-06-30 test --workspace --locked --no-fail-fast -- --test-threads=1` y `python -I -B scripts/tests/test_zero_wait_gate.py -v`.
Expected: todos los resultados verdes; V2 incluido, sin filtros ni reintentos selectivos.

- [ ] **Step 2: Ejecutar replay/shadow y Demo sólo cuando el entorno esté autorizado**

Expected: periodo/dataset/commit, ROI/PnL/WR antes y después de costes, Sharpe y EV/trade. Si no hay entorno, esos campos quedan `N/D`.

- [ ] **Step 3: Revisar OOS y estabilidad sin perseguir la meta con leverage**

Expected: evidencia neta OOS y análisis de multiplicidad; +100 %/72 h se marca confirmado, refutado o `N/D` por insuficiencia.

- [ ] **Step 4: Publicar con attestation válida y observar CI del SHA**

Expected: push normal aprobado por hook y workflows verdes del mismo SHA. No `--no-verify`, CI heredado ni Producción automática.

- [ ] **Step 5: Retirar sólo ramas que cumplan todas las guardas de Task 3**

Expected: rescate previo, worktree libre y referencias verificadas; si alguna condición falta, se conservan.

---

## Builder-check resolutions (binding)

- Attestation covers only the committed parent tree plus explicitly named documents owned by this agent. Dirty worktree and third-party overlays are excluded unless separately manifested and reviewed.
- V2 fixture, comparator, ABI 144 and `0.110` are immutable. Any formal V2 contract revision is decided **and implemented by QRA**, never by Task 5.
- If V2 remains red, final attestation, push and economic certification stop. Non-QRA audits and local fixes may continue with independent tests and no new failures, but attestation remains false.
- Demo requires `DEMO_MAINNET_AUTHORIZATION.json` with SHA/config/period/environment/human approver. Production requires separate explicit authorization plus OOS evidence. Otherwise execution is prohibited and metrics remain `N/D`.
- Before Task 7, an OOS protocol is frozen with dataset/period, incumbent commit, real fee/funding/slippage model, walk-forward split, multiplicity and decision rule. 72 h/20 trades is only an operational minimum; EV net > 0 and log growth ≥ ln(2)/72 h are required to call the hypothesis compatible, otherwise refuted or `N/D`.
