# Revisión V40 desde la base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconciliar V39 con GLM-205/206 y evitar que un hueco de al menos dos barras se fabrique como una sola barra de 16 segundos en el examen evolutivo.

**Architecture:** Sujetos inmutables en overlays, índices Git privados y fusión por árboles con padres explícitos. Reconciliación, corrección y handoff se validan por separado; ningún PASS se transfiere entre SHA o perfiles.

**Tech Stack:** Rust nightly `2026-06-30`, Cargo/rust-lld, Git/GitHub CLI, `.agent_tx`, `GIT_INDEX_FILE=.git/index_KIMI-QUANT-SR-V40`.

**Spec:** `docs/superpowers/specs/2026-11-06-base-review-v40-design.md`

## Global Constraints

- +100 %/72 h es una hipótesis OOS neta, no una garantía ni una orden de apalancamiento. ROI/PnL/WR/Sharpe/EV ausentes = `N/D`.
- No repetir V37–V39 ni recomponer sus arreglos en este átomo. No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy, A3 ni WIP reservado.
- No Demo/Mainnet/Producción, credenciales, push o attestation. Demo exige `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige autorización separada y OOS favorable.
- No `git add -A`, reset/stash/checkout forzado, force-push, `--no-verify`, borrado por antigüedad/tamaño ni limpieza de overlays con trabajo no resuelto.
- Sin topes de RAM/SSD/hilos ni semáforos/colas de admisión. Zero-wait overlays, `CARGO_BUILD_JOBS=16`, `rust-lld` y único `CARGO_TARGET_DIR=C:/Users/jhona/Documents/Proyectos/Trader Gemini/target`; paralelismo sólo con superficies de escritura no coincidentes.
- Todo cambio de producción exige RED→GREEN, regresión y revisión independiente. CAS de código bloqueado por baseline Clippy roja; CAS documental no implica push ni CI.

## Review Focus

- `dt == 2 * WF_BAR_MS`: debe reiniciar referencia y devolver `None`, no `Some(precio / p0 - 1.0)`; test nuevo en Task 1.
- `WF_BAR_MS < dt < 2 * WF_BAR_MS`: conserva el retorno observado sin rechazo prematuro; caracterización nueva en Task 1.
- `dt > 4 * WF_BAR_MS`: conserva el reinicio existente; `cl27_la_barra_del_examen_es_el_retorno_del_precio_por_reloj` en Task 1.
- Precio NaN/no positivo: conserva estado y no emite retorno; test CL27 existente en Task 1.
- Reloj retrogrado: `saturating_sub` no debe fabricar una barra ni perder el estado previo; caracterización nueva en Task 1.

---

### Task 0: Reconciliar V39 con GLM-205/206

**Files:**
- Read: refs Git, `AGENT_SYNC.md`, `COORDINACION_CODEX_2026-09-28.md`, `.agents/MEMORIA.md`
- Create: `.agent_tx/staging/KIMI-QUANT-SR-V40/plan-v40/subject/AGENT_SYNC.md`
- Create: `.agent_tx/staging/KIMI-QUANT-SR-V40/plan-v40/subject/COORDINACION_CODEX_2026-09-28.md`
- Create: `.agent_tx/staging/KIMI-QUANT-SR-V40/plan-v40/subject/docs/{ENV_OVERLAY_A3.md,RUNBOOK_CORRIDA_DEMO_A3_2026-11-06.md}`

**Interfaces:**
- Consumes: `f7d69533be36c5c67d55a851735b84dd2f8233d4` y `0ce05fdae5a4f215d185d86eb020c60fe812c21e`.
- Produces: commit `96d27840e5cc1e3e2c45b44350d6d499cf0bc389`, tree `dad8c00ab8789015aca2b2d7d5c8b88810069731`, con ambos commits como padres y cuatro documentos modificados respecto al padre local.

- [x] **Step 1: Clasificar refs, CI, agentes y trabajo concurrente**

Expected: remoto sólo `main`; 95 ahead/2 behind; 696 rutas WIP no usadas como fuente. Dos ramas Claude sin commits exclusivos pero con WIP, AGY con trabajo exclusivo y dos detached preservados. Unit `42149507213` y Replay `42149507195` SUCCESS en `02b389421`.

- [x] **Step 2: Resolver la única colisión documental**

Expected: unión `V39` seguida de GLM-205 en `COORDINACION_CODEX_2026-09-28.md`; GLM-206/runbook importados; `AGENT_SYNC.md` auto-fusionado y heartbeat V40 añadido. Ningún delimitador real de conflicto al inicio de línea; citas históricas preservadas.

- [x] **Step 3: Validar el archivo inmutable del candidato**

Run: `cargo test -p god-engine-core --test evidencia_ignicion_contract --locked -- --test-threads=1` y `cargo check --workspace --all-targets --locked` en `.agent_tx/staging/KIMI-QUANT-SR-V40/reconcile-v40/subject`.
Expected: 5/5 y workspace exit 0 en 7m25s. El primer intento terminó por timeout de compilación antes de ejecutar tests; el reintento usó una ventana acotada de 1800 s y logs separados.

- [x] **Step 4: CAS de reconciliación con expected SHA**

Expected: sólo después de Step 3, `git update-ref refs/heads/main 96d27840e5cc1e3e2c45b44350d6d499cf0bc389 f7d69533be36c5c67d55a851735b84dd2f8233d4`. Verificar padres, `diff --check` y cuatro archivos. Sin push ni cambio de `origin/main`.

---

### Task 1: Impedir compresión de huecos de al menos dos barras

**Files:**
- Modify: `crates/evolution-engine/src/online_daemon.rs:376-404` (implementación y comentario)
- Test: `crates/evolution-engine/src/online_daemon.rs`, módulo de tests junto a `cl27_la_barra_del_examen_es_el_retorno_del_precio_por_reloj`

**Interfaces:**
- Consumes: `barra_de_mercado(ultimo: Option<(u64, f64)>, ahora_ms: u64, precio: f64) -> (Option<(u64, f64)>, Option<f64>)`, `WF_BAR_MS = 16_000` y el candidato `96d27840`.
- Produces: la misma firma pública; reinicio con `(Some((ahora_ms, precio)), None)` cuando `dt >= 2 * WF_BAR_MS`. Ningún cambio en IDs, fees, series, backoff, replay o contratos reservados.

- [x] **Step 1: Materializar el sujeto y añadir RED y caracterizaciones**

Copiar el commit `96d27840` a `.agent_tx/staging/KIMI-QUANT-SR-V40/cl27-gap-v1/subject` sin mezclar el workspace. Añadir:

```rust
#[test]
fn cl27_un_hueco_de_dos_barras_no_se_comprime_en_una() {
    let t0 = 1_000_000u64;
    let ahora = t0 + 2 * WF_BAR_MS;
    let (estado, retorno) = barra_de_mercado(Some((t0, 100.0)), ahora, 101.0);
    assert_eq!(estado, Some((ahora, 101.0)));
    assert!(retorno.is_none(), "32 s no pueden convertirse en una barra de 16 s: {retorno:?}");
}

#[test]
fn cl27_una_lectura_tardia_menor_de_dos_barras_conserva_el_retorno() {
    let t0 = 1_000_000u64;
    let ahora = t0 + WF_BAR_MS + WF_BAR_MS / 2;
    let (estado, retorno) = barra_de_mercado(Some((t0, 100.0)), ahora, 101.0);
    assert_eq!(estado, Some((ahora, 101.0)));
    assert_eq!(retorno, Some(0.01));
}

#[test]
fn cl27_un_reloj_retrogrado_no_emite_ni_pierde_el_estado() {
    let previo = (2_000_000u64, 100.0);
    assert_eq!(barra_de_mercado(Some(previo), 1_000_000, 101.0), (Some(previo), None));
}
```

- [x] **Step 2: Ejecutar RED por el mecanismo exacto**

Run: `cargo test -p evolution-engine cl27_un_hueco_de_dos_barras_no_se_comprime_en_una -- --exact --nocapture`.
Expected: FAIL porque `retorno == Some(0.010000000000000009)`. Las otras dos pruebas pasan antes y después; no aceptar errores de compilación, fixtures o selección de cero tests como RED.

- [x] **Step 3: Implementar la guarda mínima y actualizar el comentario**

Cambiar únicamente `} else if dt > 4 * WF_BAR_MS {` por `} else if dt >= 2 * WF_BAR_MS {`. Documentar que un hueco de **al menos dos barras** reinicia la referencia. No cambiar la firma, IDs, fees ni el resto del examen.

- [x] **Step 4: Ejecutar GREEN y regresión CL27**

Run: repetir el test exacto y después `cargo test -p evolution-engine cl27_ -- --nocapture`.
Expected: nuevo test PASS; caracterizaciones y CL27 existentes PASS, 0 ignored/filtered.

- [x] **Step 5: Ejecutar toda la crate y contratos vecinos**

Run: `cargo test -p evolution-engine --locked --no-fail-fast -- --test-threads=1` y `cargo test -p evolution-engine --test wf_numerario_invariancia_contract --locked`.
Expected: 84 pruebas de biblioteca (81 base + RED + dos caracterizaciones) y 123 de integración, **207/207**; numerario 2/2; cero fallos/ignored/filtered.

- [x] **Step 6: Workspace, formato y Clippy**

Run: `cargo check --workspace --all-targets --locked` y `cargo clippy -p evolution-engine --lib --tests --no-deps --locked -- -D warnings`.
Expected: workspace exit 0 en 4m33s; Clippy conserva baseline real 36 lib/43 lib-test, `ADDED=[]`, `REMOVED=[]` por mensaje+archivo. `rustfmt --check` sólo como diagnóstico: no aplicar formato global a `online_daemon.rs`; whitespace del diff limpio.

- [x] **Step 7: Revisión independiente y paquete de rescate**

Expected: agent-51 `Ready to merge? Yes`, sin Critical/Important. Minor documental corregido: tres pruebas nuevas y total 207. Parche y `SHA256SUMS` verificados. CAS de código sigue bloqueado por baseline.

---

### Task 2: Handoff, recursos y fases posteriores

**Files:**
- Update: `.agent_tx/diagnostics/base-review-v40-status.json`
- Update: `AGENT_SYNC.md`
- Update: `COORDINACION_CODEX_2026-09-28.md`
- Update: `.agents/MEMORIA.md`
- Update: `.superpowers/sdd/2026-11-06-base-review-v40/progress.md`

**Interfaces:**
- Consumes: Tasks 0–1, revisión independiente, parche y hashes.
- Produces: handoff recuperable con Minor diferidos y comandos reales.

- [x] **Step 1: Revisar diff y alinear impacto**

Expected: un archivo de producción, sin formato colateral ni WIP ajeno. Documentar que la corrección es de fidelidad temporal del examen y no evidencia de rentabilidad. Minor documental corregido: 84 biblioteca +123 integración =207.

- [x] **Step 2: Registrar recursos y coordinación zero-wait**

Expected: diagnóstico sin topes ni colas, target único, lanes de escritura no coincidentes, dependencias duplicadas como censo no destructivo y ausencia de purgas. Ramas/worktrees con WIP permanecen salvo retirada con todas las guardas.

- [x] **Step 3: CAS documental con índice privado**

Expected: staging explícito sólo de documentos propios, `git diff --cached --check`, commit con padres/expected SHA correctos. No CAS de código, push o attestation. Si `origin/main` cambió, queda para reconciliación separada.

- [x] **Step 4: Prerregistrar la decisión económica futura**

Expected: antes de medir, fijar dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla compatible/refutada/insuficiente para +100 %/72 h. Reportar ROI/PnL/WR antes y después de fees, Sharpe, EV/trade y periodos exactos o `N/D`. Sin autorizaciones no hay Demo/Producción.
