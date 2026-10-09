# PLAN RONDA 7 — BARRIDO EXHAUSTIVO DESDE LA BASE, ARCHIVO POR ARCHIVO

Fecha: 2026-10-09 · Autor: Qoder (Quant Sr. / línea A del consejo) · Base
auditada: **`ab240abd`** (= `origin/main`, incluye AGY Ω40–Ω45 y Ola 73).

Mandato del operador (literal, vigente): «Vuelve a iniciar otra revisión
desde la base, han cambiado muchas cosas», «recorra en una serie de fases
hasta pasar por todos los archivos uno por uno», «revisa todo, el plan y sus
fases, el código línea por línea y función por función, los resultados, la
integración línea por línea y commit por commit, Git (commits, push, merged,
que todos los cambios de verdad llegaron a main, elimina las ramas ya
mergeadas), la documentación siempre actualizada y real para sincronizar con
los demás agentes, y los recursos (memoria, disco, CPU, GPU, red) evitando
usar recursos innecesarios», «sin descartar nada pero tampoco repetir
trabajo, verificar el estado de todo», «siempre ejecutando pruebas de
comportamiento, compilación y resultados», «pensar como un universo
multivariante continuo temporal espectral, no como scalping/swing».

---

## 0. Estado de partida verificado (no asumido)

**Censo exacto contra `ab240abd`** (medido con `git ls-files` / `wc -l`):

| magnitud | valor |
|---|---|
| crates en `crates/` | 23 |
| archivos `.rs` versionados | 488 |
| líneas Rust totales | 151 290 |
| archivos en `tests/` (contratos) | 154 |
| binarios (`src/bin`, `crates/*/src/bin`) | 37 |
| `.md` versionados | 193 |
| anotaciones `#[allow(dead_code)]` | 32 en 22 archivos |

**Lo que YA está cerrado y NO se repite** (se re-verifica por ancla, no se
re-auditó desde cero):

- Ronda 2 (G0–G2): 5/5 HIGH + todos los MED drenados (olas 62–67, Ω10–Ω14).
- Ronda 3 (H0–H2): 2/2 HIGH + 7/7 MED drenados (Ω15–Ω16, olas 65–66, GLM 103/112).
- Ronda 4 (A–C): 2/2 HIGH + 6/6 MED drenados (olas 68–69).
- Ronda 5: 12/12 drenados (olas 70–71, Ω21–Ω22).
- Ronda 6: 44 hallazgos; **38 cerrados** entre Ola 73 (R6-A3/A4/A5/B1/B2/B13),
  AGY Ω41–Ω45 (A1/C1, A2, C2/A9, C3, C4, A8, A10, A12, A14, A6, A7/C7, C11,
  C12, B3–B8, B9, B10, B11, B15, B17, C5, C6, C9, C10) y deduplicación.
- Fases F0–F8 del plan maestro Codex/AGY (barrido por capa) — ver
  `docs/PLAN_AUDITORIA_BASE_2026-10-08.md` y `docs/PLAN_MAESTRO_SINCRONIZACION.md`.

**Cola R6 REAL (pre-verificada contra `ab240abd` hoy, con ancla exacta):**

| hallazgo | estado | ancla re-grepada |
|---|---|---|
| R6-A11 [LOW] doble implementación Hodge (`risk-engine/src/hodge.rs` vs `feature-engine/src/hodge_flow.rs`) | ABIERTO | `crates/risk-engine/src/hodge.rs` vive, `lib.rs` lo re-exporta |
| R6-A13 [LOW] guard `coin_id < ym_currents.len()` → `0.0` ambiguo | RESIDUAL SEMÁNTICO | `crates/god-engine-core/src/lib.rs:5107` (el índice ya es fiel: `currents[i]` usa el índice original, Ω41) |
| R6-B12 [LOW] «RLS» de β es LMS con gain fijo, sin matriz **P** ni olvido | ABIERTO | `crates/strategy-core/src/stat_arb.rs:232-236` |
| R6-B16 [LOW] camino legacy `spread_deviation` con timing post-actualización, sin caller productivo | ABIERTO | `crates/strategy-core/src/multivariate_coint.rs:214-218` |
| R6-C8 [LOW] `dbp <= dap` trivialmente cierto en libro válido | ABIERTO | `src/bin/god_engine.rs:4101`, `:4111` |
| C-02 [MED] sin productor vivo del feed spot-futuro (OU cableada pero hambrienta) | ABIERTO, zona data-ingest/host | `GlobalArena::update_spot_data` sin caller productivo |
| R6-B14 / R6-B18 [LOW/POSITIVO] diagnósticos sin defecto | SIN ACCIÓN | theta-sesgo Jensen y DSR correctos |

**Hueco de certificación detectado en main (motivación directa de esta
ronda):** AGY Ω44 y Ω45 modificaron la física del pipeline de votos
(θ fail-closed en la SDE, cobertura espectral 12 h + rampa de confianza
C^∞, Kelly continuo de Ville, gradiente LMS gauge, smoothstep C¹ del Maker)
**sin oráculo T-1 registrado** — sólo suites por crate. Mi veredicto PASA
anterior certificaba `85557469` (post-Ω43 + Ola 73), no `ab240abd`. Esta
ronda empieza con esa re-certificación en vuelo.

---

## 1. Reglas de método (invariantes del barrido)

1. **Worktree aislado** `.ronda7` (rama `qoder/ronda7-*`) para el barrido; nunca edits en el
   checkout compartido raíz si otra sesión lo usa.
2. **Auditores READ-ONLY en paralelo** por lente (integración viva /
   matemática / física), cada uno con ámbito de archivos EXPLÍCITO y
   lista de no-repetir (sección 0).
3. **Docs antes y después**: la ASIGNACIÓN se publica ANTES de ejecutar
   (reduce colisiones — lección de la 5ª/6ª convergencia); el CIERRE con
   recibos después.
4. Un hallazgo sólo se cierra con: contrato RED→GREEN en el MISMO commit,
   `cargo check --workspace --all-targets` 0 errores, y **oráculo T-1** si
   toca el pipeline de votos/riesgo/ejecución (regla: push sólo si
   PASA ≥ 11,0 %; lista sensible canónica
   `[1,10,11,17,18,20,24,27,32,33,68,69,129,130,131,141]`).
5. **Paridad sombra↔vivo en el mismo commit** (patrón «dos caras sin
   reconciliar», confirmado 6 veces).
6. `git add` sólo de archivos propios; `git add -A` PROHIBIDO; commits
   atómicos con prefijo de bloque; tras merge entre sesiones: diff contra
   CADA padre + `--all-targets`.
7. RE-GREP de toda ancla (función/campo/binario) antes de cablearla.
8. Cero trading, cero entrenamiento, cero promoción de modelos, cero
   borrado de ramas ajenas durante el barrido.

---

## 2. Fases R0–R9 (cobertura total, archivo por archivo)

| fase | ámbito (crates/archivos) | lente | entregable + prueba |
|---|---|---|---|
| **R0** | censo versionado completo + `scripts/audit_inventory.py` (reutilizado con atribución a Codex) | inventario | `docs/audit/LEDGER_RONDA7.tsv/.json`: estado por ruta (`inventariado / leído / auditado / con-contrato / con-recibo`); prueba: `generate/check/delta` 488/488. **CERRADO 2026-10-09**: `docs/audit/LEDGER_RONDA7_2026-10-09.json` (1 512 rutas versionadas, no sólo las 488 `.rs`; `check` verde sobre `534e7980`) + `docs/audit/CENSUS_CODE_MUERTO_RONDA7_2026-10-09.tsv` (203 fichas) producido por `scripts/ronda7_dead_census.py`; fichas y alcance honesto en `docs/BARRIDO_EXHAUSTIVO_FASES.md` §R7-4 |
| **R1** | `AGENTS.md`, `.agents/MEMORIA.md`, `docs/PLAN_MAESTRO_QUANT_SR_2026-10-05.md`, `docs/PLAN_MAESTRO_SINCRONIZACION.md`, `docs/ARQUITECTURA_VIVA.md`, `docs/adr/*` | metas/conceptos | tabla teoría→código→prueba; grep de restos `scalp\|swing` con veredicto de si son semántica viva o etiqueta muerta; prueba: contrato de nomenclatura |
| **R2** | `risk-engine/src/{selection_stats,ville_e_process,cramer_lundberg,ruin,drawdown,correlation_guard,leverage_matrix,orchestrator,veto_registry}.rs`, `feature-engine/src/{multifractal,lead_lag}.rs`, `quantum-arena/src/{temporal_spectrum,spectral_tape,spectral_regime}.rs` | matemática/estadística | hallazgos `R7-B*`; prueba: `cargo test -p risk-engine -p feature-engine -p quantum-arena` |
| **R3** | `risk-engine/src/hodge.rs`, `feature-engine/src/hodge_flow.rs`, `strategy-core/src/{yang_mills_gauge,stat_arb,vecm_arbitrage,multivariate_coint,maker}.rs`, `signal-engine/src/{soliton_wave,supersonic_shockwave,quantum_oscillator,coaxial_breakout,stochastic_resonance,hawkes_bessel,flow_impulse,renyi_tsallis_entropy,nash_equilibrium,perceptron_gate,trend_runner,conformal_reversion_filter}.rs` | física/cuántica | hallazgos `R7-C*`; prueba: contratos `hodge_flow_contract`, `yang_mills_gauge_contract`, `hodge_yang_mills_consensus_contract` |
| **R4** | `god-engine-core/src/lib.rs` ( entero, por bloques), `src/bin/god_engine.rs` (host), `signal-engine/src/orchestrator.rs`, `quantum-arena/src/{state,position}.rs` | núcleo vivo | hallazgos `R7-A*`; prueba: `cargo test -p god-engine-core --lib` + suites |
| **R5** | `risk-engine` (23 archivos) + `execution-engine` ({entry_dispatch,executor,router,simulator,user_data_stream,shadow}.rs) | dinero/ejecución | prueba: `cargo test -p execution-engine -p risk-engine --all-targets` |
| **R6** | `evolution-engine` ({lib,online_daemon,return_evidence,fitness,genome,genome_store,darwin}.rs), `backtest-engine` ({lib,booktick_replay,continuous_evolution_backtest}.rs), `dark-alpha-engine` | aprender/medir | prueba: `cargo test -p evolution-engine -p backtest-engine -p dark-alpha-engine` |
| **R7** | `data-pipeline`, `data-ingest`, `storage-engine`, `omniscient-registry`, `telemetry-server`, `telemetry-engine`, `flight-recorder`, `os-guardian`, `audit-engine`, `graph-*` | datos/telemetría/guardianes | prueba: suites por crate + **C-02** (productor del feed spot) como hallazgo de esta fase |
| **R8** | paridad BT↔vivo + contratos de integración + suite completa | integración | `cargo test --workspace --all-targets` + `bt_vivo_parity_audit` + oráculo T-1 sobre el árbol final |
| **R9** | recursos + Git + honestidad documental (ver §3 y §4) | operaciones | informe con números medidos; verificación de ancestría por OID; matriz claims→recibos |

Regla de cobertura: **un archivo se declara auditado sólo si tiene entrada
en el ledger con (a) lectura completa, (b) hallazgo o «sin hallazgo» explícito,
(c) prueba que lo certifica o explicación de por qué no aplica.** Un
`contract test verde` que no ejercita la física del archivo NO cuenta como
certificación (patrón R6-A7/C7).

---

## 3. Auditoría de recursos (medido hoy, 2026-10-09)

**Disco** (C: 929,7 GB total / **125,2 GB libres**, 86,5 % usado):

| ruta | MB | naturaleza |
|---|---|---|
| `target/` (compartido raíz) | 169 978 | regenerable; `debug/deps` 148 294, `debug/incremental` 8 005, `release` 13 125 |
| `.antigravity/target/debug` | 52 576 | **huérfano**: `.antigravity` NO está registrado como worktree ni tiene `.git` (sólo `target/`, 4 `.md`) — último write 2026-10-09 01:25 |
| `.ola73/` | 16 204 | mi worktree (incremental 3 752) — conserva recibos |
| `.sol-plan-2026-10-07/` | 8 902 | worktree SOL registrado |
| `C:/Users/jhona/.codex/worktrees` | 20 934 | 11 worktrees Codex (ajenos, no tocar) |
| `data/` | 24 501 | tapes/parquet del operador — NO es cache, no se borra |
| `.git/` | 2 919 | historia versionada |
| `graphify-out/` | 504 | artefactos de grafo |

**RAM**: 23,4 GB total; **libre 4,6 GB → 3,6 GB** entre las dos pasadas de
hoy (presión creciente mientras el oráculo avanza); `Memory Compression`
1 494 → 1 117 MB. Consumidores: VS Code/`Code` ×3 (≈5,1 GB), Antigravity IDE
565 MB, Qoder 540 MB.

**CPU**: **97 % → 54 %** entre pasadas. El binario `t1_cobertura_genetica`
acumuló 1 230 s de CPU en un hilo; el ETA del T-1 se infla por contención con
los IDE de las otras sesiones.

**GPU**: sólo iGPU AMD Radeon integrada sin VRAM dedicada reportada; ningún
componente del repo la invierte (inferencia en CPU, `dark-alpha`/bosques con
buffers de stack). No hay trabajo de GPU en la agenda.

**Red**: 57 conexiones TCP establecidas no-loopback (IDEs/servicios). **El
motor vivo NO está corriendo**: cero procesos `god_engine`. Consecuencia
operativa: hoy todo resultado es de backtest/contratos; no hay feed WS ni
orden despachada. No lo relanzo sin confirmar PID+StartTime y sin instrucción
expresa (regla de sesiones concurrentes).

**Política propuesta** (no ejecutada sin autorización, son caches ajenos o
compartidos):
1. `target/debug/incremental` (8 GB raíz + 3,75 GB `.ola73`) es lo único que
   se limpia sin provocar recompilación de dependencias → hacerlo entre olas,
   nunca durante un oráculo.
2. `.antigravity/target/debug` **51 GB**: AVISO a AGY/dueño en el buzón;
   si confirma que no tiene build vivo, es el mayor recuperable del repo.
3. Oráculos largos: `CARGO_BUILD_JOBS=-j2` para no saturar los 4,6 GB libres.
4. Worktrees cerrados: retirar la rama y dejar el `target` sólo si hay
   recibos que conservara (el mío: `.ola73`, lo borro al final de Ronda 7).

---

## 4. Verificación Git (medida hoy)

- `origin/main` = `ab240abd`; `main` local idéntico (ahead 0).
- Ancestría confirmada **EN_MAIN**: `724f8c9f` (Ω41), `4cc83ce4` (Ω42),
  `01ac0bd5` (Ω43), `727b993e` (Ω44), `c49516e3` (Ω45), `8b0daf01` (Ola 73
  código), `f239ba5b`/`85557469` (merges), `751db4d2`+`32b38d96` (cierre
  documental), `ab240abd` (veredicto T-1).
- Ramas remotas: **sólo `origin/main`** ⇒ no hay ramas mergeadas que borrar
  en remoto (las mías ya se borraron al cerrar Ola 73).
- Ramas locales: `main` (en main) y `codex/integration-recovery-2026-10-07`
  (**3 commits exclusivos, NO mergeada — ajenа, no se toca**).
- Worktrees registrados: raíz + 11 Codex (detached/ramas) + `.ola73` (mío) +
  `.sol-plan-2026-10-07`. `.antigravity` aparece como directorio NO
  registrado (ver §3).

---

## 5. Asignación y no-colisión

| ola | dueño | contenido | oráculo |
|---|---|---|---|
| **Ola 74 (Qoder)** | R6-B12 (β: LMS→RLS real con **P** y olvido, o renombrar y documentar) + R6-B16 (camino legacy `spread_deviation`: eliminar o cablear con contrato) + R6-A13 (guard ambiguo → clave explícita `ym_current_absent`) | `strategy-core` + ancla en core | SÍ (toca votos) |
| **R7-R0/R1 (Qoder)** | ledger de cobertura + auditoría doctrina/nomenclatura | docs + `scripts/` | NO (docs-only) |
| **C-02 (AGY o Codex)** | productor vivo del feed spot-futuro en host + paridad BT + contrato escritor↔lector con spot real | `data-ingest`/host/`quantum-arena` | SÍ |
| **R6-A11 (AGY)** | unificar las dos implementaciones Hodge (flujo por pares dirigido como patrón canónico) | `risk-engine`/`feature-engine` | SÍ |
| **R6-C8 (Qoder, cola)** | discriminante real de presión de libro (no `dbp <= dap` trivial) | host | SÍ |
| **Re-certificación Ω44/Ω45** | Qoder (**PASA 16/144 = 11,1 %**, 1 748,60 s sobre `534e7980`≡código de `ab240abd`) | T-1 sobre `ab240abd` | es el oráculo |

AGY/GLM/Codex/SOL: **no duplicar** lo de la tabla; si alguien toma un
hallazgo de esta cola, que lo marque en el buzón ANTES de ejecutar.

---

## 6. Bitácora

- 2026-10-09: plan publicado (esta ola). Censo, recursos, Git y cola R6
  medidos contra `ab240abd`, no recordados. Oráculo T-1 de re-certificación
  de main post-Ω44/Ω45 **CERRADO: PASA 16/144 = 11,1 % ≥ trinquete 11,0 %**
  (exit 0, 1 748,60 s, lista sensible idéntica a la canónica). Veredicto
  completo con su evidencia de árbol: `docs/BARRIDO_EXHAUSTIVO_FASES.md` §R7-3.
- 2026-10-09: **R0 CERRADA** (ledger de 1 512 rutas + censo de 203 fichas +
  escáner versionado) → §R7-4 del BARRIDO.
