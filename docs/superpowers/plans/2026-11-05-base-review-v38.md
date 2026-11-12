# Revisión V38 desde la base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconciliar referencias y trabajo concurrente, corregir un defecto libre con evidencia causal y dejar una revisión V38 recuperable sin fabricar certificación económica.

**Architecture:** Overlays inmutables por sujeto, índices privados Git y fusión por árboles con padres explícitos. La implementación es atómica: primero propiedad y RED, después cambio mínimo, regresión, revisión independiente y sólo entonces cualquier CAS permitido.

**Tech Stack:** Rust workspace nightly `2026-06-30`, Cargo/rust-lld, Git/GitHub CLI, Graphify existente, `.agent_tx` y `GIT_INDEX_FILE` privado.

**Spec:** `docs/superpowers/specs/2026-11-05-base-review-v38-design.md`

## Global Constraints

- +100 %/72 h es hipótesis OOS neta, no garantía ni instrucción de apalancamiento. ROI/PnL/WR/Sharpe/EV ausentes = `N/D`.
- No repetir V37: PR #171, GLM-203, backoff ni censo config/lib. No transferir PASS entre SHA, overlays, perfiles o toolchains.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy ni WIP ajeno. No `git add -A`, reset/stash/checkout forzado, force-push o `--no-verify`.
- No Demo/Mainnet/Producción, credenciales, push o attestation. Demo exige `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige autorización separada y OOS favorable.
- No borrar por antigüedad/tamaño. Cada retirada exige valor/ancestría, cero commits exclusivos, worktree libre, archivos preservados y ausencia de reserva.
- Todo cambio de producción exige RED→GREEN y revisión independiente. `check` no equivale a prueba, CI ni attestation.
- Overlays `.agent_tx/staging/KIMI-QUANT-SR-V38/`, índice `.git/index_KIMI-QUANT-SR-V38`, nightly `2026-06-30`, `CARGO_BUILD_JOBS=16` y único `CARGO_TARGET_DIR` canónico.

## Review Focus

- Una rama ancestro con worktree sucio o archivos necesarios: conservar y rescatar, nunca forzar borrado (Task 1).
- Un commit exclusivo que parece viejo: revisar valor antes de integrar o eliminar; la antigüedad no prueba abandono (Task 1).
- Un literal, fallback o teoría sin llamador vivo: no constituye por sí solo un defecto ni justifica producción (Task 2).
- Un test que pasa antes del cambio por razones equivocadas: RED debe fallar por el comportamiento objetivo, no por fixture roto (Task 2).
- Un PASS previo transferido a otra fuente: prohibido; repetir evidencia sobre el árbol final (Tasks 2 y 4).
- RAM/SSD bajos: no purgar trabajo ni detener procesos ajenos; medir y acotar sólo con evidencia (Tasks 0 y 4).

---

### Task 0: Corte de verdad, diseño e inventario

**Files:**
- Read: `AGENT_SYNC.md`, `COORDINACION_CODEX_2026-09-28.md`, `.agents/MEMORIA.md`
- Create: `docs/superpowers/specs/2026-11-05-base-review-v38-design.md`
- Create: `.agent_tx/diagnostics/base-review-v38-status.json`
- Create: `.agent_tx/diagnostics/v38-inventory.json`
- Create: `.agent_tx/diagnostics/v38-inventory.tsv`

**Interfaces:**
- Consumes: `HEAD=ba27c367749dae443a488006f6b79b03d7b2941e`, `origin/main=6f211252fce278b0c47a0d95c0f7c3724ce1f814`.
- Produces: diseño con builder `nothing important left` e inventario mecánico de 852 rutas Rust, sin afirmar revisión semántica.

- [x] **Step 1: Reconciliar refs, CI, agentes y recursos**

Run: `git branch -a`, `git ls-remote --heads origin`, `git worktree list --porcelain`, `gh run list --branch main --limit 8` y medición CIM de recursos.
Expected: sólo `main` remota; Unit/Replay verdes; 1,55 GiB RAM libres y 159,66 GiB SSD libres, sin acción destructiva.

- [x] **Step 2: Registrar heartbeat y diseño con builder independiente**

Expected: heartbeat V38; enfoque verificación primero; builder agent-44 reporta `nothing important left`.

- [x] **Step 3: Crear inventario mecánico por fases**

Run: `git ls-files '*.rs'` y clasificación por subsistema.
Expected: 852 rutas con `INVENTORIADO`, `UNASSIGNED`, `N/D`; no `REVISADO` inferido.

---

### Task 1: Clasificar y, si procede, retirar referencias fusionadas

**Files:**
- Read: ramas `agy/wave3-omega220`, `claude/linea-i-i14`, `claude/qsr-lc20r2`, `claude/qsr-lcf020` y sus worktrees.
- Update: `.agent_tx/diagnostics/base-review-v38-status.json`

**Interfaces:**
- Consumes: HEAD/orígenes de Task 0 y revisión independiente agent-42.
- Produces: decisión por rama `DELETE_ONLY_WITH_GUARDS`, `PRESERVE` o `RESCUE`, con evidencia.

- [x] **Step 1: Verificar valor, PR, ancestro, commits exclusivos y suciedad**

Run por rama: `git rev-list --count main..<branch>`, `git merge-base --is-ancestor <branch> main`, `gh pr list --state all --head <branch>` y estado del worktree.
Expected: agent-42 confirma que `linea-i-i14`, `qsr-lc20r2` y `qsr-lcf020` tienen cero commits exclusivos y son ancestros; `agy/wave3-omega220` tiene uno (`3ff61ae3`) y se preserva. Los dos worktrees detached y los estados sucios de Línea I/QSR-R2 quedan preservados.

- [x] **Step 2: Aplicar guardas de worktree y archivos**

Expected: retirar únicamente `claude/qsr-lcf020` y `C:/Users/jhona/Documents/Proyectos/tg-claude-qsr-lc20` si el estado sigue limpio, PR #163 está `MERGED`, el blob ignorado `target/qsr-cl07-remote.rs` coincide con el contrato preservado y no cambió ninguna guarda. Cualquier incumplimiento ⇒ `PRESERVE` con causa; no `remove --force`.

- [x] **Step 3: Ejecutar retirada permitida y verificar**

Run sólo con todas las guardas: `git worktree remove C:/Users/jhona/Documents/Proyectos/tg-claude-qsr-lc20`, `git branch -d claude/qsr-lcf020` y `git worktree prune`; no existe rama remota que eliminar.
Expected: sólo `qsr-lcf020` se retira. `linea-i-i14`, `qsr-lc20r2`, `agy/wave3-omega220` y ambos worktrees detached permanecen por WIP/valor. `git branch -a` y `git ls-remote --heads origin` confirman el estado final.

---

### Task 2: Suprimir reentrenamiento de una revisión fallida sin datos nuevos

**Files:**
- Modify: `crates/evolution-engine/src/online_daemon.rs`
- Test: `crates/evolution-engine/tests/online_persist_backoff_contract.rs`
- Update: `.agent_tx/diagnostics/base-review-v38-status.json`

**Interfaces:**
- Consumes: candidato V37 `persist-backoff-v1`, copiado a `.agent_tx/staging/KIMI-QUANT-SR-V38/retrain-revision-v1/subject`; `forest.observation_revision() -> u64` y `run_online_learning_loop` en `src/bin/god_engine.rs:2946`.
- Produces: `LiveEvolutionDaemon.retrain_failed_revision: Option<u64>`; `should_retrain_forest_now(pending_new_obs, obs_count) -> bool` conserva la pierna periódica y la inhibición por persistencia, pero exige una revisión no intentada en la pierna pendiente.

- [x] **Step 1: Materializar el candidato V37 actual sin mezclar main**

Run: copiar `persist-backoff-v1/subject` a `.agent_tx/staging/KIMI-QUANT-SR-V38/retrain-revision-v1/subject`.
Expected: fuentes idénticas al candidato V37; no se usa el árbol antiguo `upstream-merge-v37` ni el WIP compartido.

- [x] **Step 2: Escribir RED de revisión fallida sin rearme y nueva revisión que sí rearma**

Añadir `fallo_de_entrenamiento_espera_una_revision_nueva_sin_borrar_pending` con 50 observaciones de una sola clase y `iteration_count=1`. Después del fallo, la misma revisión debe devolver `false` con `pending_new_obs=10`; una nueva observación aceptada debe devolver `true` con 11/51.

- [x] **Step 3: Ejecutar RED y comprobar la causa**

Run: `cargo test -p evolution-engine --test online_persist_backoff_contract --locked -- --nocapture`.
Expected: falla la aserción posterior a `RetrainFailed`: la revisión no cambió pero el disparador pendiente sigue verdadero; no se exige API nueva ni fixture inválido.

- [x] **Step 4: Implementar la revisión de fallo mínima**

Añadir `retrain_failed_revision: Option<u64>`, inicializarlo a `None`, guardarlo sólo en `RetrainFailed` y limpiarlo tras éxito. En la pierna pendiente usar `pending_new_obs >= 10 && self.retrain_failed_revision != Some(self.forest.observation_revision())`. No modificar `online_random_forest.rs`, `pending_new_obs`, el disparador periódico ni la deuda de persistencia.

- [x] **Step 5: Ejecutar contratos y regresión completa**

Run: `cargo test -p evolution-engine --test online_persist_backoff_contract --test online_forest_persistence_contract --test online_forest_hardening_contract --test online_training_determinism_contract --locked`.
Expected: 6/6 backoff, 6/6 persistencia, 8/8 hardening y 2/2 determinismo, cero ignored/filtered. Después `cargo test -p evolution-engine --locked --no-fail-fast -- --test-threads=1`: 235/235.

- [x] **Step 6: Workspace, Clippy y revisión independiente**

Run: `cargo check --workspace --all-targets --locked` y `cargo clippy -p evolution-engine --lib --tests --no-deps --locked -- -D warnings`.
Expected: workspace exit 0; Clippy conserva baseline 33/40 sin diagnóstico nuevo. Revisión independiente obligatoria; CAS de código permanece bloqueado por baseline roja.

---

### Task 3: Revisión, handoff y fases posteriores

**Files:**
- Update: `.agent_tx/diagnostics/base-review-v38-status.json`
- Update: `AGENT_SYNC.md`
- Update: `COORDINACION_CODEX_2026-09-28.md`
- Update: `.agents/MEMORIA.md`

**Interfaces:**
- Consumes: Tasks 0–2, revisión independiente y resultados reales.
- Produces: handoff recuperable con bloqueos, Minor diferidos y comandos.

- [x] **Step 1: Revisar diff contra sujeto y padres**

Expected: ningún archivo ajeno ni formato colateral. Critical/Important exige RED→GREEN; Minor queda en ledger con coste explícito.

- [x] **Step 2: CAS documental con índice privado**

Expected: sólo documentos propios, `git diff --check` y expected SHA. No CAS de código con baseline roja ni push/attestation.

- [x] **Step 3: Prerregistrar la fase OOS antes de cualquier corrida futura**

Expected: dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión fijados antes de medir. Sin autorizaciones no hay Demo/Producción; métricas o `N/D`.

---

## Registro de cierre ejecutado

- Task 0 completo: heartbeat, builder `nothing important left`, inventario mecánico de 852 rutas Rust.
- Task 1 completo: sólo `qsr-lcf020` retirado con todas las guardas; cualquier otro trabajo permanece preservado, incluido `tg-glm204` aparecido después.
- Task 2 completo: RED 5/6 → GREEN 6/6; full 235/235; workspace 3m36s; Clippy baseline 40/40 sin delta. Revisión agent-45 `Ready to merge? Yes`, sin Critical/Important.
- Task 3: handoff documental con índice privado y sin push/attestation. El CAS de código sigue bloqueado por baseline.
- Minor diferido: cobertura de éxito posterior y de excepción periódica. No se amplía producción para cubrirlo en esta ola.

El candidato de rescate es `.agent_tx/staging/KIMI-QUANT-SR-V38/retrain-revision-v1/retrain-revision.patch`. El próximo agente debe revalidar fuentes y hashes, no rehacer el átomo ni transferir PASS a otro SHA. Producción NO-GO; métricas económicas `N/D`.
