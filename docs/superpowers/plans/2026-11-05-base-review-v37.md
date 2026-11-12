# Revisión V37 desde la base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar el PR #171 con evidencia completa y cerrar el defecto de persistencia que puede reentrenar cada 500 ms, sin mezclar sujetos ni fabricar resultados económicos.

**Architecture:** Validación por árboles y overlays inmutables, fusión Git con padres explícitos y un átomo backoff independiente. Config/lib se audita por llamadores reales antes de cualquier cambio; Demo/OOS queda detrás de autorización y propietarios.

**Tech Stack:** Rust workspace nightly `2026-06-30`, Cargo/rust-lld, bincode, Git/GitHub CLI, `.agent_tx` y `GIT_INDEX_FILE` privado.

**Spec:** `docs/superpowers/specs/2026-11-05-base-review-v37-design.md`

## Global Constraints

- +100 %/72 h es hipótesis OOS, nunca garantía ni instrucción de apalancamiento. ROI/PnL/WR/Sharpe/EV ausentes = `N/D`.
- V2 conserva fixture, comparador, ABI 144 y mínimo `0.110`; QRA decide e implementa cualquier revisión.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, legacy ni WIP ajeno. No `git add -A`, reset/stash/checkout forzado, force-push, `--no-verify` o purgas.
- No Demo/Mainnet/Producción, lectura de credenciales, kills, borrado de overlays/worktrees o limpieza por tamaño/antigüedad. Demo exige `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige otra autorización y OOS.
- Todo cambio de producción exige RED→GREEN y revisión independiente. `check` no equivale a prueba ni attestation.
- Un único `CARGO_TARGET_DIR` canónico, overlays `.agent_tx/staging/KIMI-QUANT-SR-V35/` e índice absoluto `.git/index_KIMI-QUANT-SR-V35`.

## Review Focus

- CI rojo/cancelado: CAS local y borrado deben detenerse; no se presenta integración como completada (Task 0).
- Guardado que falla una vez o repetidamente: sólo `save_snapshot` se reintenta; no se reentrena si las observaciones no cambiaron (Task 2).
- Observaciones durante el backoff: se conservan y producen exactamente una nueva generación tras guardar la pendiente (Task 2).
- `RetrainFailed` no programa reintento; `PersistFailed` mantiene la deuda hasta persistir, con espera máxima 30 s y sin límite de intentos (Task 2).
- Ausencia o valor fijo en config/lib: sólo se corrige si un llamador demuestra defecto; ausencia = `N/D`, nunca otro literal (Task 3).

---

### Task 0: Cerrar PR #171 localmente y condicionar referencias

**Files:**
- Read: `.agent_tx/diagnostics/pr171-local-merge-tree.txt`
- Read: `.agent_tx/staging/KIMI-QUANT-SR-V35/pr171-merged/i13-test.log`
- Read: `.agent_tx/staging/KIMI-QUANT-SR-V35/pr171-merged/workspace-check.log`
- Update: `.agent_tx/diagnostics/base-review-v37-status.json`

**Interfaces:**
- Consumes: tree `4e1d6fb5f23cc20119641796dbeb6ebc0384834f`, merge commit `c4db0cdd48dc0fe031fb0fbe181ee58ff1d7a2e1`, padres `fb780dd4` y `8493e48e`.
- Produces: CAS local sólo con CI verde y worktree limpio; sin CAS ni borrado si CI es rojo/cancelado.

- [x] **Step 1: Ejecutar contrato I-13 y workspace del árbol**

Run: `cargo test -p trader-gemini-v5 --test i13_sucesor_cuota_contract --locked` y `cargo check --workspace --all-targets --locked` sobre `pr171-merged/subject`.
Expected: 4/4 y `Finished dev profile` en 3m39s, ambos exit 0.

- [ ] **Step 2: Observar CI del merge hasta resultado completo**

Run: `gh run list --commit 8493e48e1832a6d6505dccae70a0a4fa204fba9e`.
Expected: `Replay contracts` y `Unit and contract suites` SUCCESS. Si falla o se cancela, no hay CAS ni borrado; recomponer desde `fb780dd4`.

- [ ] **Step 3: Aplicar CAS local con padres verificados**

Run: `git update-ref refs/heads/main c4db0cdd48dc0fe031fb0fbe181ee58ff1d7a2e1 fb780dd46fecc3ea323586a28a299ac22ec82763`.
Expected: `HEAD=c4db0cdd`, padres `fb780dd4` y `8493e48e`, tree `4e1d6fb5`; `git diff --check` exit 0.

- [ ] **Step 4: Retirar referencias sólo si todas las guardas pasan**

Run: `git worktree remove C:/Users/jhona/Documents/Proyectos/tg-linea-i-i13`, `git branch -D claude/linea-i-i13`, `git push origin --delete claude/linea-i-i13`.
Expected: sólo después de PR `MERGED`, CI verde y worktree limpio. Cualquier suciedad detiene la retirada y exige rescate. Las demás ramas/worktrees permanecen intactos.

---

### Task 1: Publicar diseño, plan y estado sincronizado

**Files:**
- Create: `docs/superpowers/specs/2026-11-05-base-review-v37-design.md`
- Create: `docs/superpowers/plans/2026-11-05-base-review-v37.md`
- Update: `.agent_tx/diagnostics/base-review-v37-status.json`
- Update: `AGENT_SYNC.md`
- Update: `COORDINACION_CODEX_2026-09-28.md`

**Interfaces:**
- Consumes: Task 0 y builder-check con cuatro resoluciones vinculantes.
- Produces: plan canónico, heartbeat y buzón recuperables por cualquier agente.

- [x] **Step 1: Resolver las cuatro preguntas del builder**

Expected: tree vs commit, autorización de retirada, CI como puerta y semántica completa del backoff quedan en `## Resoluciones del builder check (vinculantes)`.

- [ ] **Step 2: Registrar heartbeat y buzón V37**

Expected: refs reales, CI actual, validación I-13/workspace y RED de backoff. Ningún CI se declara antes del resultado. Producción NO-GO y métricas `N/D`.

- [ ] **Step 3: Crear árbol documental con índice privado y CAS**

Run: leer `HEAD`/`origin/main`, escribir sólo los dos documentos de coordinación más los archivos propios del plan y `git diff --check`.
Expected: WIP e índice canónico intactos; CAS documental sin push ni attestation. Si cambian refs, recomponer antes del CAS.

---

### Task 2: Error tipado y backoff de persistencia

**Files:**
- Modify: `crates/evolution-engine/src/online_daemon.rs`
- Modify: `crates/evolution-engine/src/online_random_forest.rs`
- Create: `crates/evolution-engine/tests/online_persist_backoff_contract.rs`
- Update: `.agent_tx/diagnostics/base-review-v37-status.json`

**Interfaces:**
- Consumes: overlay independiente `persist-backoff-v1/subject`, no el árbol de PR #171.
- Produces:

```rust
pub enum ForestPersistError { RetrainFailed(String), PersistFailed(String) }
pub struct ForestPersistRetry {
    next_attempt: Instant,
    delay: Duration,
    observation_revision_at_failure: u64,
}
impl ForestPersistRetry {
    pub fn new(now: Instant) -> Self;
    pub fn next_attempt(&self) -> Instant;
    pub fn delay(&self) -> Duration;
    pub fn is_ready(&self, now: Instant) -> bool;
    pub fn register_failure(&mut self, now: Instant);
    pub fn clear(&mut self);
}
pub enum ForestPersistRetryOutcome { Saved, Retrained }
pub fn retry_forest_persist(&mut self, now: Instant) -> Option<Result<ForestPersistRetryOutcome, String>>;
pub fn retrain_forest_and_persist(&mut self) -> Result<(f64, f64), ForestPersistError>;
pub fn should_retrain_forest_now(&self, pending_new_obs: usize, obs_count: usize) -> bool;
pub fn observation_revision(&self) -> u64;
```

- `forest_persist_retry: Option<ForestPersistRetry>` se añade al daemon.
- `register_failure` duplica `delay` hasta 30 s y fija `next_attempt=now+delay`; `clear` restaura 500 ms. Los intentos continúan hasta éxito.
- Con retry presente, `should_retrain_forest_now` devuelve `false` aunque `obs_count>=10 && iteration_count%20==0` o `pending_new_obs>=10`.
- `observation_revision` es monótona, vive sólo en memoria y se incrementa al aceptar cada observación válida; no cambia `TGFOREST` ni se reinicia al reentrenar. La comparación por `len()` queda prohibida porque el drenaje la hace disminuir.
- Recuperación: guarda primero la generación pendiente; si `observation_revision()>observation_revision_at_failure`, reentrena y guarda una sola generación nueva. `pending_new_obs` se reinicia sólo cuando el snapshot final contiene todas las observaciones.

- [x] **Step 1: Escribir el RED**

Crear tres contratos: `fallo_de_persistencia_reintenta_solo_guardado_y_recupera_sin_reentrenar`, `backoff_es_determinista_acotado_y_se_reinicia_tras_exito` y `fallo_de_entrenamiento_no_programa_reintento_de_persistencia`.

- [x] **Step 2: Ejecutar RED**

Run: `cargo test -p evolution-engine --test online_persist_backoff_contract --locked -- --nocapture`.
Expected: siete errores por tipos/métodos/campo ausentes en `persist-backoff-v1/red.log`; no es un fixture roto.

- [ ] **Step 3: Ampliar RED con observaciones durante backoff y ambos disparadores**

Añadir `observaciones_durante_backoff_generan_una_sola_nueva_generacion` y `ambos_disparadores_de_reentrenamiento_se_inhiben_con_deuda_pendiente`.
Expected: antes de implementar, no existe la API y el test falla. Tras ella: generación antigua+1 y snapshot completo; retry presente ⇒ ambos false, retry ausente ⇒ condiciones originales.

- [ ] **Step 4: Implementar tipos y máquina de estados mínima**

Implementar las firmas de Interfaces inicializando `forest_persist_retry=None` en `LiveEvolutionDaemon::new`, sin alterar warmup, restauración ni snapshot `TGFOREST`. `retrain_forest_and_persist` mapea fallos a `RetrainFailed`/`PersistFailed`; `retry_forest_persist` sólo guarda salvo observaciones nuevas detectadas por revisión monótona. Ningún `.unwrap()`/`.expect()` nuevo en producción.

- [ ] **Step 5: Ejecutar contratos y suite completa**

Run: `cargo test -p evolution-engine --test online_persist_backoff_contract --test online_forest_persistence_contract --test online_forest_hardening_contract --test online_training_determinism_contract --locked`.
Expected: 3/3 nuevos contratos más 2/2 ampliados, 6/6, 8/8 y 2/2 respectivamente. Después `cargo test -p evolution-engine --locked --no-fail-fast -- --test-threads=1`: 234/234, 0 ignored/filtered.

- [ ] **Step 6: Workspace, Clippy y revisión independiente**

Run: `cargo check --workspace --all-targets --locked` y `cargo clippy -p evolution-engine --lib --tests --no-deps --locked -- -D warnings`.
Expected: workspace exit 0. Clippy puede conservar 33/40 baseline y 5 de feature-engine, pero ningún diagnóstico nuevo. Revisión independiente obligatoria; CAS de código sigue bloqueado por baseline.

---

### Task 3: Censo config/lib sin fabricar un cambio

**Files:**
- Read: `crates/quantum-arena/src/config.rs`
- Read: `crates/quantum-arena/src/lib.rs`
- Read: `src/lib.rs`
- Create: `.agent_tx/diagnostics/config-lib-v37-census.json`

**Interfaces:**
- Consumes: llamadores reales y `qsr_cl07_drawdown_consejo_contract` para `CANONICAL_COUNCIL_MAX_DRAWDOWN`.
- Produces: cada hallazgo con archivo:línea, llamador, hipótesis y decisión `OPEN_ATOM` o `NO_CHANGE`.

- [ ] **Step 1: Censar literales, fallbacks y ausencias**

Run: búsquedas dirigidas de `13.0`, `0.0001`, `0.0002`, `0.0005`, `0.55`, `0.60`, `0.50`, `unwrap_or` y `clamp` en los tres archivos.
Expected: no se infiere defecto por presencia de un literal. `CANONICAL_COUNCIL_MAX_DRAWDOWN` se clasifica por su función actuarial y contrato, no como hardcode arbitrario.

- [ ] **Step 2: Elegir una única corrección sólo si hay evidencia**

Expected: un llamador vivo y una hipótesis causal que justifique un átomo. Si no existe, registrar `NO_CHANGE` y no crear producción ni tests para fingir progreso.

- [ ] **Step 3: Si existe un defecto, escribir RED antes de producción**

Expected: test que falle por comportamiento incorrecto, no por texto del archivo. Sin reemplazar un default válido por otro literal ni debilitar vetos justificados. Si no hay defecto, este paso queda sin aplicación.

---

### Task 4: Revisión, handoff y fases posteriores

**Files:**
- Update: `.agent_tx/diagnostics/base-review-v37-status.json`
- Update: `.agent_tx/diagnostics/config-lib-v37-census.json`
- Update: `AGENT_SYNC.md`
- Update: `COORDINACION_CODEX_2026-09-28.md`
- Update: `.superpowers/sdd/2026-11-05-base-review-v37/progress.md`

**Interfaces:**
- Consumes: Tasks 0–3 y revisión independiente.
- Produces: recibo final con bloqueos y Minor diferidos, sin certificación económica.

- [ ] **Step 1: Revisar diff contra cada padre y sujeto**

Expected: ningún archivo ajeno, formato colateral ni PASS transferido. Critical/Important exige RED→GREEN; Minor queda en ledger con coste explícito.

- [ ] **Step 2: Actualizar ledger y CAS documental**

Expected: hashes de fuentes, comandos, resultados, CI real y bloqueos. Sólo documentos propios; `git diff --check` y CAS expected SHA. No CAS de código con baseline rojo ni push/attestation.

- [ ] **Step 3: Prerregistrar OOS antes de cualquier corrida futura**

Expected: dataset/periodo, commit incumbente, fees/funding/slippage reales, partición walk-forward, multiplicidad y regla de decisión fijados antes de medir. Sin `DEMO_MAINNET_AUTHORIZATION.json`, no hay Demo; Producción exige autorización separada y OOS. ROI/PnL/WR antes/después de fees, Sharpe, EV/trade, periodo y dataset o `N/D`.

- [ ] **Step 4: Retirar referencias únicamente con todas las guardas**

Expected: PR/ancestría, cero commits exclusivos, worktree libre, archivos rescatados y ausencia de lease/reserva. Si falta una condición, se conserva. La siguiente fase de ignición B1/A4-T2 y V2 permanece con sus dueños.

---

## Registro de cierre ejecutado

- Task 0 completo: Unit/Replay SUCCESS; CAS local `c4db0cdd`; rama/worktree I-13 retirados con todas las guardas. Otros worktrees conservados.
- Task 1 completo como handoff documental: diseño/spec, plan, estado máquina y buzones preparados para CAS sin código, push ni attestation.
- Task 2 completo sin CAS de código: revisión agent-41 `With fixes`; Critical de drenaje corregido con `capacity-red.log`→`capacity-green.log`; Important de aislamiento corregido. Resultado final 234/234, workspace exit 0 y Clippy sin diagnósticos nuevos pero baseline roja.
- Task 3 completo: cuatro `NO_CHANGE`, cero `OPEN_ATOM`, sin producción. Detalle en `config-lib-v37-census.json`.
- Task 4: revisión y handoff registrados. Un Minor diferido: validación explícita del contenido del snapshot/contabilidad extraíble. El Minor documental del coste periódico fue corregido en el ledger (hasta 20 iteraciones). No se ejecutó OOS; su protocolo queda prerregistrado, no medido ni certificado.

El reemplazo canónico del átomo es `persist-backoff-v1/revision.patch`. Ningún agente debe reanudar desde `online-persistence-v1` ni aplicar la versión que comparaba `observations.len()`. Producción NO-GO; métricas económicas `N/D`.
