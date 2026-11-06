# Revisión V39 desde la base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconciliar upstream y V38, corregir un defecto libre con evidencia causal y dejar handoff recuperable sin certificación económica inventada.

**Architecture:** Overlays inmutables, índices privados Git y fusión por árboles con padres explícitos. Primero propiedad y RED, después cambio mínimo, regresión, revisión independiente y sólo entonces cualquier CAS permitido.

**Tech Stack:** Rust nightly `2026-06-30`, Cargo/rust-lld, Git/GitHub CLI, `.agent_tx` y `GIT_INDEX_FILE` privado.

**Spec:** `docs/superpowers/specs/2026-11-05-base-review-v39-design.md`

## Global Constraints

- +100 %/72 h es hipótesis OOS neta, no garantía ni instrucción de apalancamiento. ROI/PnL/WR/Sharpe/EV ausentes = `N/D`.
- No repetir V37/V38: PR #171, GLM-203/204, backoff ni censo config/lib. No transferir PASS entre SHA, overlays o perfiles.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy ni WIP ajeno. No `git add -A`, reset/stash/checkout forzado, force-push o `--no-verify`.
- No Demo/Mainnet/Producción, credenciales, push o attestation. Demo exige `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige autorización separada y OOS favorable.
- No borrar por antigüedad/tamaño. Cada retirada exige valor/ancestría, cero commits exclusivos, worktree libre, archivos preservados y ausencia de reserva.
- Todo cambio de producción exige RED→GREEN y revisión independiente. `check` no equivale a test, CI ni attestation.
- Overlays `.agent_tx/staging/KIMI-QUANT-SR-V39/`, índice `.git/index_KIMI-QUANT-SR-V39`, nightly `2026-06-30`, `CARGO_BUILD_JOBS=16` y único `CARGO_TARGET_DIR`.

## Review Focus

- Upstream verde pero conflicto documental: unión sin marcadores y diff contra cada padre antes de CAS (Task 0).
- Rama/worktree con WIP o commit exclusivo: conservar; antigüedad no demuestra abandono (Task 0).
- Literal, fallback o teoría sin llamador vivo: no justifica cambio (Task 1).
- RED que falla por fixture roto en vez del comportamiento objetivo: inválido; debe probar el defecto real (Task 1).
- PASS transferido a otro SHA: prohibido; repetir evidencia sobre árbol final (Tasks 1 y 3).
- RAM/SSD bajos: no purgar trabajo ni detener procesos ajenos (Tasks 0 y 3).

---

### Task 0: Corte de verdad y reconciliación upstream

**Files:**
- Read: `AGENT_SYNC.md`, `COORDINACION_CODEX_2026-09-28.md`, `.agents/MEMORIA.md`
- Create: `docs/superpowers/specs/2026-11-05-base-review-v39-design.md`
- Create: `.agent_tx/diagnostics/base-review-v39-status.json`

**Interfaces:**
- Consumes: V38 `4d3712d4` y upstream `02b389421`/`7afc9ed1`.
- Produces: merge `62df45d9bf53fb88438cc03dc293e3c0d3f3202b`, tree `e4e2696310f1844590f18c1d7fa6f5f843b2950e`, con `origin/main` como ancestro.

- [x] **Step 1: Reconciliar refs, CI, agentes y ramas**

Run: `git branch -a`, `git ls-remote --heads origin`, `git worktree list --porcelain`, `gh run list --branch main` y fetch de `origin/main`.
Expected: sólo `main` remota; upstream GLM-204 Unit/Replay SUCCESS. WIP y ramas con valor preservados.

- [x] **Step 2: Unir upstream y validar**

Run: `git merge-tree --write-tree HEAD origin/main`, unión documental y `cargo test -p god-engine-core --test evidencia_ignicion_contract --locked`, `cargo check --workspace --all-targets --locked`.
Expected: sin marcadores; 5/5 y workspace exit 0 en 3m39s. CAS `62df45d9`, padres `4d3712d4` y `7afc9ed1`, tree `e4e26963`.

- [x] **Step 3: Registrar diseño con builder independiente**

Expected: heartbeat V39; enfoque verificación primero; builder agent-47 `nothing important left`.

---

### Task 1: Preservar identidad de moneda en fees del examen

**Files:**
- Modify: `crates/evolution-engine/src/online_daemon.rs`
- Test: `crates/evolution-engine/tests/fee_identity_contract.rs`
- Update: `.agent_tx/diagnostics/base-review-v39-status.json`

**Interfaces:**
- Consumes: candidato V38 `retrain-revision-v1`, copiado a `.agent_tx/staging/KIMI-QUANT-SR-V39/fee-identity-v1/subject`; `forest.observation_revision` y `fee_rates_for(coin_id)` permanecen sin cambios.
- Produces: pares `(source_coin_id, returns)` durante filtrado/ordenación; `roundtrip_fees` calculado con el ID original; `wf_evaluate_real` usa esos IDs sin asignación densa.

- [x] **Step 1: Materializar el candidato V38 sin mezclar main**

Run: copiar `retrain-revision-v1/subject` a `.agent_tx/staging/KIMI-QUANT-SR-V39/fee-identity-v1/subject`.
Expected: fuentes idénticas al candidato V38; no se usa WIP compartido ni overlays anteriores.

- [x] **Step 2: Escribir RED de identidad en filtrado y orden por longitud**

Añadir `v39_fee_identity_survives_filter_and_length_sort` con tasas coin 0 `0.0001`, coin 1 `0.01`, coin 0 plano y coin 1 elegible. Extraer las dos transformaciones existentes preservando primero comportamiento: `prescreen_exam_inputs` y `ranked_exam_series`. El contrato exige fee de coin 1; para la ordenación usa IDs fuente `(1,0)` de entrada y exige `[0,1]` tras ordenar por longitud, evitando la coincidencia posición/ID del primer fixture.

- [x] **Step 3: Ejecutar RED y verificar mecanismo**

Run: `cargo test -p evolution-engine --test fee_identity_contract --locked -- --nocapture`.
Expected: `red-behavior6.log` falla por fee e IDs erróneos (`[1,0]`), no por API ausente ni fixture roto.

- [x] **Step 4: Conservar source_coin_id hasta fees y replay**

Llevar `(source_coin_id, returns)` en filtrado y ordenación. Construir fees por ID original y pasar esos IDs directamente al juez real, sin `wf_real_coin_ids` posicional. No editar otros crates ni registro global.

- [x] **Step 5: Ejecutar contratos y regresión completa**

Run: contrato nuevo 1/1; backoff 6/6, persistencia 6/6, hardening 8/8 y determinismo 2/2, cero ignored/filtered. Después `cargo test -p evolution-engine --locked --no-fail-fast -- --test-threads=1`: 236/236.

- [x] **Step 6: Workspace, Clippy y revisión independiente**

Run: `cargo check --workspace --all-targets --locked` y `cargo clippy -p evolution-engine --lib --tests --no-deps --locked -- -D warnings`.
Expected: workspace exit 0 en 3m17s; Clippy conserva baseline 33/40 con `ADDED=[]` y `REMOVED=[]`. Revisión agent-48 `Ready to merge? Yes`, sin Critical/Important. CAS de código permanece bloqueado por baseline roja.

---

### Task 2: Revisión, handoff y fases posteriores

**Files:**
- Update: `.agent_tx/diagnostics/base-review-v39-status.json`
- Update: `AGENT_SYNC.md`
- Update: `COORDINACION_CODEX_2026-09-28.md`
- Update: `.agents/MEMORIA.md`

**Interfaces:**
- Consumes: Tasks 0–1, revisión independiente y resultados reales.
- Produces: handoff recuperable con bloqueos, Minor diferidos y comandos.

- [x] **Step 1: Revisar diff contra sujeto y padres**

Expected: dos archivos autorizados, sin formato colateral. Minor diferidos: compartir la transformación ID-preservante hoy duplicada (mismo comportamiento, escaneo/clonaciones acotadas una vez por ronda y riesgo de divergencia futura) y cubrir directamente series cortas, `WF_REAL_WINDOW` y `WF_REAL_MAX_COINS` (hoy cobertura indirecta).

- [x] **Step 2: CAS documental con índice privado**

Expected: sólo documentos propios, `git diff --check` y expected SHA `62df45d9`. No CAS de código con baseline roja ni push/attestation.

- [x] **Step 3: Prerregistrar OOS antes de cualquier corrida futura**

Expected: dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión fijados antes de medir, como exige el diseño §Fase 3. Sin autorizaciones no hay Demo/Producción; métricas o `N/D`.
