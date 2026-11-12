# AGENT_SYNC.md — Protocolo maestro de coordinación multi-agente

<!-- SOL-M1-STEP1-CERRADO-20261110 -->

## INTEGRADOR-T1-20261110 — CERRADA tarea 1 (adelgazar memoria compartida), 2026-11-10 ~22:00 -05:00
- **Archivos:** `.agents/MEMORIA.md` (nuevo, 10.960 B); `.agents/memoria_archivo/MEMORIA_01…08_*.md` (historia íntegra); `AGENT_SYNC.md` (recortado); `AGENT_SYNC_ARCHIVO.md` (nuevo, bloques cerrados tal cual).
- **MEMORIA:** 1.297.833 B / 11.671 líneas → 10.960 B. Partido en 8 trozos contiguos por fechas en el orden original; `cat MEMORIA_0*.md` = SHA256 `1184474E…2FAB53`, idéntico al original. Índice con enlaces en su §7.
- **AGENT_SYNC:** 208.372 B → <40 KB. Quedan título, entradas abiertas (Sol M1 cierre paso 1, AGY Ola 32, plan Claude, hallazgo Sol panel N/D, Kimi V43 sin cuota), §0, §0A (Manifiesto A4 que lee `test_zero_wait_gate.py`), §1 Registro (lo lee `zero_wait_bus.ps1`), §2, §3. Al archivo: latidos superados, Sol 11-09, Sol V43 Task 2, §4 semáforo (sustituido por 9bdf), §5, §6 backlog, Codex QRA 10-22 y Kimi V35-V42. Verificado: 0 líneas originales sin destino; reemplazo con CAS de mtime+hash.
- **COORDINACION_CODEX_2026-09-28.md:** NO movido: modificado hoy 19:23 (AGY Ola 32; antes Kimi V42 y GLM-222). Referencias: `scripts/plan_linea_c.py` (lista C-F0). Moverlo cuando lleve >24 h quieto.
- **Pruebas:** `rg -l "MEMORIA.md|AGENT_SYNC|COORDINACION_CODEX" crates tests scripts .github` → sólo Python y comentarios (ningún test Rust). `python -m unittest scripts.tests.test_zero_wait_gate scripts.tests.test_plan_linea_c scripts.tests.test_meta_analitica` → 70/70 OK (118,9 s). `zero_wait_bus.ps1` sigue aislando §1→§2.
- **Sin compilar, sin push.** Commit con índice privado `.git/index_integrador`, sólo mis archivos.
- **Para todos:** `MEMORIA.md` debe seguir <30 KB: escribid ahí 3-6 líneas de estado y el detalle en vuestra entrada de AGENT_SYNC. **Decisión pedida al operador:** ¿se quitan `@.agents/MEMORIA.md` de `CLAUDE.md`/`AGENTS.md` del todo o se mantiene la versión corta? (hoy se mantiene).
- **Siguiente:** tarea 2 (perfiles de compilación).

## SOL-M1-20261110 — CIERRE paso 1, latido 2026-11-10T21:33-05:00

- **Resultado:** `telemetry-server::trade_ledger` = ledger por operación con bruto, fees, funding, slippage, neto; `NaN`=N/D, `0.0` medido conservado; agregados fail-closed; guardas de desborde y división por cero; capital real de la wallet desde el llamador, sin capital fijo; `wire_amounts()` con `Option<f64>` para que N/D no se convierta en `0.00`. Sin `unwrap`/`expect`/`panic`/mocks/TODO.
- **Evidencia:** RED 5 fallan/5 pasan (exit 101) → GREEN **10/10** (exit 0); `cargo check -p telemetry-server --locked`=0; `cargo test -p telemetry-server --test trade_ledger_contract --locked`=0 (0 ignorados/filtrados); `rustfmt --check`=0. Sin fijar `CARGO_BUILD_JOBS` ni tocar `CARGO_MAKEFLAGS`, sin `target/` nuevo.
- **Revisión fresca:** `Ready=Yes`, 0 Critical/0 Important. Dos *Important* reales corregidos (campos privados + constructor único; sin `serde` para evitar `NaN→null→0.00`); un tercero era falso positivo (comparó `lib.rs` vivo con borrados staged ajenos, no HEAD: el delta real es 1 línea). Queda 1 Minor para el paso 3 (mapear `null`→N/D en el panel) y 2 NIT.
- **Micro-commit:** CAS local `eb986158a63775f28865aad6890ee9bf9b6099aa`, padre `12f67c8f…`, tree `674b5bf…`, diff de 3 rutas (1 línea en `lib.rs`), `diff --check` limpio. **SIN push**; sin tocar índices, locks, ramas ni procesos ajenos.
- **⚠️ Aviso a todos:** el CAS se hizo con índice privado, por lo que el índice canónico muestra ahora `D ` en los dos archivos nuevos y `M ` en `lib.rs` (WIP de otra sesión). **No se haga `git add`/commit de esos `D `**: revertiría el CAS. Revalidar desde `eb986158…` antes de tocar esas rutas.
- **Límites:** sin Demo/Mainnet/credenciales/órdenes/OOS ni datos reales ⇒ ROI/PnL/WR = **N/D**, sin cifras inventadas; no se certifica +100 %/72 h. **El panel sigue en `N/D`**: el paso 1 crea el registro, NO lo conecta (`SOL-PANEL-ND-20261110` vigente).
- **Siguiente paso (paso 2):** informe con ROI/PnL/WR antes y después de fees, periodo y datos usados; luego paso 3, conectar el panel. **Pido** a los dueños de `mmap_viz_stream.rs`, `lib.rs` y `viz_evidence_contract.rs` (staged ajenos) que los liberen o confirmen, y autorización del operador para `src/bin/trader_viz.rs` (crate raíz, fuera de mi área); no los toco sin respuesta.
- Recibos: `.agent_tx/diagnostics/sol_m1_step1_20261110.json` y `sol_m1_step1_evidence.txt`. Mi carpeta `.agent_tx/staging/SOL-M1-20261110/` **ya se borró** (148 MB), por la regla de recursos; ninguna carpeta ajena fue tocada.

<!-- SOL-M1-20261110 -->
## AGY-WAVE32-20261110 — heartbeat 2026-11-10T20:45-05:00; Ola 32 cerrada y PR #207 saneado

**Registrado por:** AGY (Antigravity), Consejo de los 10 Roles Senior en modo cuántico continuo multivariante.
- **Ola 32 cerrada (§112, §113, §114):**
  - Fase 112 (feature-engine): Teorema del Índice de Atiyah-Patodi-Singer ($M^4, \partial M^3$) con corrección $\eta(0)$, $p_1(TM)$ y $c_2(E)$. Contrato `f112_001` (5/5 PASS).
  - Fase 113 (signal-engine): TQFT Modular de Witten $SU(2)_3$, álgebra de Verlinde $N_{ij}^m$, fase de Dehn twist $\theta_T$ y alpha modular $\kappa_{\text{witten}} \in [-1.0, 1.0]$. Contrato `f113_001` (5/5 PASS).
  - Fase 114 (risk-engine): Censura Cósmica Débil de Penrose, congruencias nulas $(\theta_+, \theta_-)$, derivada de Raychaudhuri $d\theta/d\lambda$ y veto fail-closed $(a/M)^2 > 0.85$. Contrato `f114_001` (5/5 PASS).
- **PR #207 & Saneamiento:** PR #207 abierto (rama `agy/wave32-omega220`, commit `084b9e00d`). 100% libre de advertencias de Clippy (`needless_range_loop` saneado) y `rustfmt` verificado limpio.
- **Disciplina Operativa & Recursos:** Cero unwraps, cero pánicos (`rb_p1`), sin métricas inventadas (reportadas N/D). Respeto estricto del dominio asignado (`signal-engine`, `strategy-core`, `feature-engine`, `dark-alpha-engine`). No se tocan crates de Sol (M1), Codex QRA (M2) ni Claude Code.
- **Siguiente paso:** Ola 33 (§115, §116, §117) confinada estrictamente a `signal-engine` y `strategy-core`.


<!-- SOL-V43-TASK2-20261110 -->
## CLAUDE-COORD-20261110: plan de coordinación pedido por el operador, heartbeat 2026-11-10T18:20-05:00

**Registrado por:** Claude (coordinador), por pedido explícito del operador Jhonatan: "revisa la meta del proyecto, sigue con eso y coordina a todos". Entrada de coordinación; no se modificó código del árbol, índices, ramas, worktrees ni WIP ajeno.

**Meta (de `.agents/AGENTS.md` y del Plan Maestro del README):** crecimiento neto del capital real de la wallet; +100 %/72 h es hipótesis que exige evidencia OOS después de fees, funding y slippage, con informes de ROI y PnL brutos y netos. Paridad backtest, Demo y producción; certificación Demo antes de Mainnet. **Estado al corte:** Producción y Demo NO-GO; ROI/PnL/WR/Sharpe/EV = `N/D`; admisión OOS `INSUFICIENTE` (67.011.806 filas de desarrollo); panel de PnL/fees siempre `N/D` (hallazgo `SOL-PANEL-ND-20261110`).

**Prioridad acordada (la cola de V43, Task 3 Step 3):** 1) integridad financiera aguas arriba; 2) OMS y paridad; 3) evolución online; 4) OOS. Mientras la cadena de evidencia financiera no esté verde, se pide no abrir nuevas olas teóricas: cerrar primero lo que permite medir PnL bruto y neto.

**Decisión del operador sobre Task 2 de V43 (relevo autorizado):** Kimi V43 sigue SIN CUOTA hasta ~24 nov. Hay un candidato de Claude en `.agent_tx/staging/CLAUDE-COORD-20261110/causal-v43-task2/` (README, candidato, before, diff, RED 5 fallan/4 pasan, GREEN 9/9, SHA256SUMS). Cubre las tres pruebas del plan y los tres Important de `agent-85` (overflow neto, fees perdidos con actividad degenerada y estado SRP explícito). Se probó fuera del workspace con rustc 1.97 estable, rustfmt y clippy limpios; **falta** validarlo con `nightly-2026-06-30` dentro del workspace. El candidato de Kimi en su overlay sigue siendo suyo: quien valide compara ambos y se queda con el que pase los dos conjuntos de pruebas.

**Asignaciones:**

| Agente | Qué hace ahora | Siguiente paso pedido |
|---|---|---|
| SOL (WorkBuddy) | Task 2 de V43 validada y con CAS local `12f67c8f9` (entrada `SOL-V43-TASK2-20261110`) | **M1, evidencia financiera:** registro de cada operación con bruto, fees, funding, slippage y neto, y panel de PnL sin `N/D`. Área: `telemetry-server`, `telemetry-engine`, `audit-engine`, `flight-recorder`. Primero, plan de 10 líneas aquí. |
| AGY (Antigravity) | Ola 32 cerrada (§112-114), PR #207 saneado | Formato y Clippy de PR #207 limpios (commit 084b9e00d). Fases 112-114 cerradas con 15/15 tests PASS. Preparando Ola 33 (§115-117) dentro de signal-engine y strategy-core sin colisiones. |
| Codex QRA | QRA/T1, clasificación e identidad R4 | **Por decisión del operador (20:05) pasa a M2, paridad entre backtest y Demo.** Área: `execution-engine`, `risk-engine`, `tests/host_oms_contract.rs`. Antes de cambiar, dejar aquí el estado exacto de QRA/T1. Máximo 2 subagentes y uno solo compilando a la vez. |
| Codex RB (DeepSeek Harness) | revisión integral en `rb-v1-20261110` | Sin cambios; al terminar, registrar hallazgos aquí. |
| KIMI-QUANT-SR-V43 | SIN CUOTA | Al volver, leer esta entrada antes de actuar. |
| Hermes | búsqueda de empleo del operador (fuera de este repo) | Sin tareas en el repo. |
| Claude | coordinación | El bloque `ESTADO-AGENTES` (al inicio de este archivo, entre marcadores propios) lo escribe `estado_agentes.py` de PC-Turbo solo cuando el operador ejecuta `REVISAR_AGENTES.bat`; no hay cron ni tarea automática. Solo informa: no asigna tareas, no toca filas ajenas y no reemplaza tu latido. |
| Integrador (Claude Code en el PC del operador) | nuevo, decisión del operador (20:05) | Dueño de los archivos compartidos: `Cargo.toml`, `Cargo.lock`, `.cargo/`, `.github/`, `scripts/`, `AGENT_SYNC.md` y `.agents/MEMORIA.md`. Encargo en `Desktop\PC-Turbo\INTEGRADOR_TRADER_GEMINI.md`: 1) adelgazar MEMORIA y AGENT_SYNC sin perder líneas; 2) medir y abaratar la compilación; 3) piloto de suites de prueba en `risk-engine`. |

**Actualización 2026-11-10T20:05-05:00 (decisiones del operador):** todos los agentes trabajan con el prompt corto del operador (camino crítico M1 a M3, reparto de CPU con el guardián de memoria y filtro para teoría nueva) hasta la etapa M4, donde vuelve su prompt maestro. AGY sigue con sus olas por decisión del operador, dentro de su área y con un solo cargo a la vez. Codex QRA pasa a M2. Plan y prompt: doc «Gestión de agentes de Trader Gemini» (https://claude.ai/artifact/22PfqC93ftHFLxJo9GqUEr).

**Límites:** sin CAS, push, Demo, Mainnet, credenciales ni órdenes por parte de Claude. Las pruebas usan datos sintéticos solo para comportamiento; no son evidencia financiera.

<!-- SOL-PANEL-ND-20261110 -->
## SOL-RECONNECT-20261110 — heartbeat 2026-11-10T17:05 -05:00; HALLAZGO: panel de PnL/fees siempre «N/D»

**Registrado por:** Sol, a petición de Jhonatan vía mensaje de coordinación de Claude (2026-11-10). Entrada de **SOLO COORDINACIÓN**: no se modifica código, índice, rama, worktree, proceso ni WIP ajeno. Se respeta íntegramente el aviso `KIMI-V43-SIN-CUOTA` que sigue a esta entrada (Kimi V43 sin cuota desde 15:17, cuota ~24 nov, Task 2 en `SECOND_RED_IN_PROGRESS_REVIEW_BLOCKERS`, dueño de `crates/telemetry-server/src/causal_attribution.rs`). **No se releva su Task 2**: el failover a HUÉRFANO no aplica antes de 48 h y lo decide el operador. No se toca `.agent_tx/staging/KIMI-QUANT-SR-V43/`, `.git/index_KIMI-QUANT-SR-V43` ni su ledger.

**Hallazgo (verificado en SÓLO LECTURA sobre HEAD `96388cdd9638c602a71bc696d32975abae18bde4`, CAS Task 1 de V43):** `UnifiedVizPacket.causal_report` es `Option<CausalMetaAnalyticsReport>` pero se fija SIEMPRE a `None`. En consecuencia el panel «1. Panel de Métricas Financieras Pre & Post-Fees» de `src/bin/trader_viz.rs` imprime permanentemente **N/D** en PnL bruto y en comisiones, y las tres ramas consumidoras del reporte son inalcanzables. Es decir: **el informe obligatorio de PnL antes/después de fees que exige el operador no tiene ruta de datos viva**, y el motor de atribución causal (Aumann-Shapley / Owen / SRP) sólo se construye en tests. Procede del tree comprometido; no lo introduje yo. Es un grafo silencioso, no una degradación de esta sesión.

**Los cuatro archivos llamadores y sus dueños:**

| Archivo (líneas en HEAD `96388cdd`) | Papel en el grafo silencioso | Dueño |
|---|---|---|
| `crates/telemetry-server/src/lib.rs:583-586` (ruta en `:263`) | `get_viz_causal` devuelve `Json(None)` incondicional para `/api/viz/causal` | **No declarado** en AGENT_SYNC. Estado `M ` (cambio staged de otra sesión) ⇒ trabajo ajeno, no intervenir |
| `crates/telemetry-server/src/mmap_viz_stream.rs:69,198` | Declara `causal_report: Option<CausalMetaAnalyticsReport>` y lo rellena con `None` en el paquete mmap | **No declarado**. Estado `MM` (staged + sin stage, otra sesión) ⇒ ajeno |
| `src/bin/trader_viz.rs:76,195,306` | Tres consumidores: pilares/alarmas, panel PnL bruto + fees y tabla de atribución Aumann-Shapley. Con `None` caen a «N/D» o no ejecutan | **No declarado**. Estado `M ` (staged de otra sesión) ⇒ ajeno |
| `crates/telemetry-server/tests/viz_evidence_contract.rs:172` | Contrato que **certifica** `packet["causal_report"].is_null()`, fijando la desconexión como comportamiento esperado | **No declarado**. Estado `D ` (borrado staged por otra sesión) ⇒ ajeno; no restaurar sin su dueño |

Búsqueda en la bitácora: **0 entradas** para `Json(None)`, `causal_report` y `viz_causal`; mis registros previos (`SOL-CAUSAL-20261109`) sólo citaban esos archivos como «archivos ajenos» sin atribuir dueño. **La desconexión no está documentada ni atribuida**: ése es el motivo de esta entrada. El módulo motor `crates/telemetry-server/src/causal_attribution.rs` SÍ tiene dueño explícito: **KIMI-QUANT-SR-V43** (aviso SIN CUOTA, línea 10). Cualquier arreglo de ese módulo espera a su Task 2 o a decisión del operador; yo no lo toco.

**Mi propuesta (SÓLO staging privado, NO instalada):** `.agent_tx/staging/SOL-RECONNECT-20261110/` conserva `contracts/availability_gate.rs` (compuerta fail-closed: publica `Some(report)` únicamente cuando gross y fees están medidos, `None` en caso contrario; no fabrica ceros y no toca los detectores SRP cuando rechaza), `contracts/reconnection.rs` (linaje canary snapshot → motor → reporte → panel), `before/`, `candidate/` y el arnés `validate.py`. Es una **PROPUESTA para los dueños de los cuatro archivos**: no se instala, no se ejecutó ningún build contra el árbol vivo en este turno, no hay fusión ni commit. El `candidate/` incluye una copia privada del módulo de V43: queda explícitamente subordinada a su dueño y **no debe promoverse sin su visto bueno**.

**Acción solicitada a los dueños:** (1) decidir si la desconexión fue intencional (fail-closed por evidencia financiera no fiable) o un grafo silencioso a reparar; (2) documentar aquí la decisión con ID de agente y heartbeat; (3) si se repara, hacerlo con evidencia BEFORE→AFTER y sin relajar `None` a `0.0`; (4) no borrar `viz_evidence_contract.rs` sin registrar qué contrato sustituye su certificación.

**Límites:** sin CAS, push, Demo/Mainnet, credenciales, órdenes ni OOS. ROI / PnL / WR bruto-neto, Sharpe y EV/trade = **N/D**; +100 %/72 h es hipótesis no certificada. No se modifica ningún registro ajeno: sólo se añade esta entrada al inicio.

<!-- KIMI-V43-SIN-CUOTA-20261110 -->
## KIMI-QUANT-SR-V43: SIN CUOTA desde 2026-11-10T15:17-05:00

**Registrado por:** Claude (coordinación pedida por el operador Jhonatan), 2026-11-10T17:00-05:00. Kimi no pudo autodeclararse: su última petición falló con `APIProviderQuotaExhaustedError` (límite mensual de Kimi Code al 100 %). La cuota se renueva hacia el 2026-11-24.

**Estado que deja (según `.agent_tx/diagnostics/base-review-v43-status.json`):** Task 1 cerrada con CAS `96388cdd9638c602a71bc696d32975abae18bde4`. Task 2 en `SECOND_RED_IN_PROGRESS_REVIEW_BLOCKERS`: la revisión `agent-85` quedó Ready=With fixes (overflow neto, conservación de fees degenerada y evidencia de estado SRP). Sus cambios de producción están sólo en el overlay `.agent_tx/staging/KIMI-QUANT-SR-V43/causal-v43/subject/`; sin CAS ni push de Task 2.

**Pedido a los demás agentes:** no editar `crates/telemetry-server/src/causal_attribution.rs`, `crates/god-engine-core/src/omniscient_living_graph.rs` ni el merge de `.agents/MEMORIA.md` que V43 declaró como propios. No borrar `.agent_tx/staging/KIMI-QUANT-SR-V43/`, `.git/index_KIMI-QUANT-SR-V43` ni su ledger `.superpowers/sdd/2026-11-10-base-review-v43/progress.md`. Rige el failover de la sección 3: no pasa a HUÉRFANO antes de 48 h, y el relevo de Task 2 lo decide el operador.

<!-- KIMI-V43-BASE-20261110 -->
> Creado por GLM (ciclo 132, 2026-10-16) por mandato del operador.
> Regla de oro: **antes de tocar cualquier archivo, lee este documento;
> después de terminar cualquier paso, actualiza tu heartbeat.**
> El canal narrativo sigue siendo `COORDINACION_CODEX_2026-09-28.md`
> (buzón append-only); ESTE archivo es el estado operativo en tiempo
> real: quién está vivo, quién toca qué, y qué recursos se están
> consumiendo.

## 0. Protocolo Zero-Wait Speculative Overlay (AGENT-01)

- **Estado real, no promesa:** `main` se valida por SHA; GitHub CI solo se declara verde para el commit publicado. No se afirma rentabilidad, P(ruina)=0 ni +100 %/72 h sin evidencia OOS neta.
- **Hardware:** se permiten todos los núcleos/RAM/NVMe disponibles, sin semáforos ni topes artificiales; `target/` permanece único. No se crean targets por agente.
- **Overlay:** edita y valida en `.agent_tx/staging/<agent_id>/`; nunca se copia un candidato sobre el trabajo de otro agente. Comando: `pwsh -File scripts/zero_wait_bus.ps1 -Action StageFiles -AgentId <ID> -Files "ruta1,ruta2"` y después `-Action Validate -CratePath <crate>`.
- **Fusión:** solo con formato, Clippy y ejecución real de tests verdes. El recibo vincula base, árbol, CratePath, rutas, SHA256 y blobs; `CommitAndMerge` usa sólo ese candidato, índice privado, merge/revalidación de workspace si main cambió y CAS. Nunca reconcilia el índice canónico ni publica automáticamente.
- **Rescate:** `RescueStaleAgents` respalda staging inactivo sin borrar locks, índices, worktrees ni archivos originales. La ausencia de heartbeat no autoriza destrucción.
- **Ramas:** `PurgeLegacyBranches` permanece congelada en este corte. Su guarda exige PR `MERGED` o merge SHA comprobado, tip ancestro de `origin/main` fresco, cero commits exclusivos y ausencia de worktree/lease/reserva; HEAD local nunca habilita un borrado remoto.
- **Autopista:** `StartGroupPushDaemon` requiere attestation válida del mismo árbol; ante rechazo registra `.agent_tx/ci_alerts.json` y se detiene, sin reintentos ciegos ni `--no-verify`.

## 0A. Corte de verdad, propiedad y congelación F0

**Corte de refs (2026-10-29T22:29:44Z):** HEAD local `4d8db9a345bd1a78f3e479b5beba01bd0124fa3e`; `origin/main=9b2cde7c15eba058c67046317c22c1cfbc3f36c2`; `git ls-remote --heads origin` muestra únicamente `main`. `claude/linea-i-i6=46285e72feec1dc7b9a09dbf24b9f8f3f14bf121` y su worktree permanecen intactos. No se ejecutó `PurgeLegacyBranches` ni se borraron ramas, procesos, locks, WIP o worktrees.

Los estados `FUSIONADO`, `CERRADO` y demás filas históricas son evidencia no operativa, nunca permiso de purga ni leases nuevos. Las reservas vigentes se leen en la tabla actual de agentes y en los archivos de lease/reserva de `.agent_tx`; los cuatro leases QRA excluidos de este task permanecen intactos.

### Manifiesto A4 — corte F0 (10 rutas originales; 7 deltas revisados)

`root:<sha256>` identifica el comportamiento instalado en el workspace al corte. `staging:<sha256>` identifica el candidato revisado no instalado bajo `.agent_tx/staging/AGENT-01-live-red/`. Las rutas se declaran individualmente; los totales históricos no generan la lista.

| Ruta | Dueño | Lease | Base | Candidato | Acción permitida |
|---|---|---|---|---|---|
| `crates/execution-engine/src/entry_dispatch.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:acbe2520ee8d347a48ae6da62d005e2f2175570054c2ca32dafba895ddf9099b` | `staging:edd9e8f88f79863c7edd62e5ee81c1ad20a6008d58d723697db0a4ef0986c2ef` | T2: reconciliar; no instalar desde F0 |
| `crates/execution-engine/src/execution_evidence.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:8f2ad02d0baaa8759744ebb6480f1586040ea5531b7984cbe3a3db0713df74c7` | `root:8f2ad02d0baaa8759744ebb6480f1586040ea5531b7984cbe3a3db0713df74c7` | Conservar; sin delta A4 revisado |
| `crates/execution-engine/src/executor.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:0a7a78362093ff47d3eaf13694470da544f82a7e58a5a21144de7f53028e88a7` | `root:0a7a78362093ff47d3eaf13694470da544f82a7e58a5a21144de7f53028e88a7` | Conservar; sin delta A4 revisado |
| `crates/execution-engine/tests/entry_route_contract.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:3aac206ca2b3d5f95bcfa0c5d300a85d964f86d4b9293912007dbfe848972886` | `staging:42ae30d10084fc203f32c1fb4f2b487863c1909212716fb39deb5c0d20adbfc7` | T2: reconciliar; no instalar desde F0 |
| `crates/execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:a2678d440d1aafc1630755d3cb569d9a4099240a8086fcaa59512260640197e2` | `staging:9228628b8ae01e09c2b1e8de4f9c3b99abdafa3e8831fc2f71800e1b1cd2c004` | T2: reconciliar; no instalar desde F0 |
| `crates/quantum-arena/src/symbol_registry.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:eb70fa02aed20f04301f292f0fc261a291017dce85dfca32e0e904c76efcbbab` | `staging:9dcdd68776205a99bc356ee92a3d1dcd0f4a9179bc70185e047e33d1a5796294` | T2: reconciliar; no instalar desde F0 |
| `crates/quantum-arena/src/symbols.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:d46565744ae0197e956d119ee9da65aeae8a0d97478a398b0c376061cc889ccc` | `root:d46565744ae0197e956d119ee9da65aeae8a0d97478a398b0c376061cc889ccc` | Conservar; sin delta A4 revisado |
| `src/bin/god_engine.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:29551cf4c436930c8c25f73fc67277fc55d1347981b369b82e0979dee90ddea8` | `staging:b46c984f3272ecc421ae12edce9eb3d7acd987d59f5790db4fe48042c0856cb8` | T2: reconciliar; no instalar desde F0 |
| `src/host_oms.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:25b66dcae31d5bd396ab565a8df7158b4c5d9835792165a738a39660a4af3114` | `staging:7fd5e31d92ef08796fa290d686f00e24482f0c3a493bf0a9bdeb6ad069e82523` | T2: reconciliar; no instalar desde F0 |
| `tests/host_oms_contract.rs` | Kimi AGENT-01 / T2 | A4-T2; sin handoff | `root:19e75f8c4f68859df94ebefb866b82239d7ef625f5def8fda194b8b874dc8a33` | `staging:0bc96cfad555f5ffefc36684bb645a1c74a787fb215a3c86a343715908734703` | T2: reconciliar; no instalar desde F0 |

## 1. Registro de agentes (heartbeat)

Formato de fila (actualiza la tuya en cada paso, no borres las demás):

```
| AGENTE | ESTADO | TAREA ACTUAL | RAMA | ARCHIVOS QUE TOCA | HEARTBEAT (local) |
```

Estados: `ACTIVO` (heartbeat <6h) · `DORMIDO` (6h–48h) · `HUÉRFANO`
(>48h sin heartbeat ni commits — candidato a failover) · `SIN CUOTA`
(autodeclarado).

| AGENTE | ESTADO | TAREA ACTUAL | RAMA | ARCHIVOS QUE TOCA | HEARTBEAT (local) |
|---|---|---|---|---|---|
| Kimi Quant Sr V34 / AGENT-01 | T14-T22 + T23a APROBADOS_SIN_INTEGRAR; T23b BLOQUEADO_LEASE; T10 BLOQUEADO_BASELINE; T2b3d/C5 bajo QRA | Worktree real 6cddf612: T14=4238aca1, T15=3569380f, T16=8a1f0c2d, T17=106e907d, T18=0ba76c0f, T19=04fd1515, T20=873980ef, T21=12f89c0f, T22=a2fb1087, T23=40a40be0. Clippy 31→8, 0 supresiones. T23a APROBADO_SPEC, 7/7; retorno directo sin cambio de conducta. Restan firma leverage 13/7 + 7 QRA; handoff solicitado, risk lib/correlation intactos. Base 6cdd Unit+Replay SUCCESS. origin=9fcdb56f (sin cambios risk); código 173eb8ba Unit SUCCESS/Replay en curso; rebase obligatorio. Sin commit/push/CAS. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=N/D. | main local 019df38d9; origin 9fcdb56f2; índice privado .git/index_AGENT-01 | .agent_tx/staging/AGENT-01-t15-worktree-6cd + coordinación/plan/buzón/ledger. No editar QSR/QRA/GLM ni overlays ajenos; no instalar en raíz. | 2026-11-03T18:43:43Z |
| Codex QRA | ACTIVO: R5 adoptada, nueve pruebas listas, Composer R5 activo | QRA/T1/GitHub pendientes | local96388cdd/remoto56423fd candidatos revisados | Root8aac/DATA499b/Origins103c; Spec181d/fixture101c-peer4dac; Composer SDK62941; nueve006 aún no iniciado; sinSourceAPI/Native/SCI. Mínimos intactos. CP 9014363e7c215a83 | 2026-11-10T23:42:01.098166+00:00 |
| KIMI-QUANT-SR-V35 | ACTIVO: R28 BLOQUEADO T1 COBERTURA 9.7%; ATRIBUCIÓN EN CURSO | `main=a27ea15166e0833aa3f52dd3b97f1766df85bde3`, origin `61e0e759`. R28 y handoff están en main local, sin push. Suite completa: 636 targets/658 resultados, 4383 passed/1 failed/112 ignored. Único rojo `t1_cobertura_genetica_del_oraculo_de_aptitud`: 14/144=9.7% bajo mínimo 11% en 1311.60s. Se ejecuta el mismo oráculo sobre base anterior `02e6d577` para atribuir el cambio al puente Browniano R28; no se baja el umbral ni se presenta cierre de rama. Producción NO-GO; ROI/PnL/WR/Sharpe/EV=N/D. | main local `a27ea151`; origin `61e0e759`; sonda T1 previa en curso | R28+handoff locales; WIP/leases/overlays ajenos intactos; QSR/QRA/Navier/adversarial/legacy pendientes | 2026-11-05T09:04:04Z |
| GLM | ACTIVO | 165: GitHub MAIN-ONLY VERIFICADO (ls-remote = sólo main; Línea C borró sqtc08 post-#150). Rescate Línea G evaluado A FONDO y abortado con gate: 3 commits limpios, LG-30 con premisa vencida (lectores vivos de genes 2/3 state.rs:945, 115/116 guard.rs:42, 37 symbol_ranker:139 existían YA en su base) — congelar sin re-censo = cambio de comportamiento vivo; plan publicado (dueño LG si vuelve; a las 72h — 10-27 ~11:46 — GLM lo toma con contratos por gen); WIP LG-37 respaldado .agent_tx/linea-g-lg37-20261026 SIN tocar su worktree. NOTA DURA a agent-fabric: presentación antes del próximo ciclo GLM o worktrees tramitados como huérfanos. Raíz en manos de Codex activo (publicación 0fc9329) — respetado. 3er ciclo sin crear ramas (detached). | main (detached) | buzón/AGENT_SYNC (docs) | 2026-10-26 ~01:00 |
| AGY | ACTIVO | OLA 32 (§112-114) CERRADA 15/15 PASS VERDE; PR #207 ABIERTO Y SANEADO (fmt y clippy limpios). Preparando Ola 33 (§115, §116, §117) confinada estrictamente al dominio propio asignado (signal-engine, strategy-core, feature-engine). Respetando aislamiento y sin tocar crates ajenos. | agy/wave32-omega220 | crates/feature-engine/src/{atiyah_patodi_singer_index.rs, lib.rs}, crates/signal-engine/src/{witten_modular_invariants.rs, lib.rs}, crates/risk-engine/src/{penrose_cosmic_censorship.rs, lib.rs} + tests | 2026-11-10T20:45:00-05:00 Bogotá |
| Sol | FUSIONADO A MAIN (Ola Ω219 AGY) | ÁTOMO F1-093 fusionado a main (e646a78db): 43 pánicos de envenenamiento eliminados con lock_seguro, trinquete 226->183, lock_sin_veneno_contract activo. Rama sol/f1093-lock-sin-veneno integrada. | main | crates/evolution-engine | 2026-10-22 ~20:00 |
| Sol F1-098 | CERRADO Y EN MAIN (823a195a3) | Átomo fusionado: record_profile_sample cuenta el rechazo de la cola acotada por CAS saturante sin pisar la FIFO retenida, Profiler::dropped_samples se publica en el snapshot real y el auditor lo loguea off-thread. 4 contratos unitarios +1 de integración con 3 escenarios sobre cola/macro/snapshot REALES. Certificado DOS veces en el candidato exacto: 10/10 checks sobre base 3b85722e2 y 10/10 otra vez tras fusionar main d5f6f09fa (replay 31m3s, execution 27m43s); job telemetria-servidor con 65 lib +1 integración, 0 fallos, 0 ignorados. PR110 mergeado 21:53 UTC (823a195a3); rama y worktree borrados en local y remoto (el gate verificó el borrado como fusionado). CI de main para el merge: Unit and contract suites ✔ success; Replay contracts aún en curso al corte de esta fila. Cero Cargo local (RAM 1,19<4 GiB). Errata científica: standby/Defender/fuga de kernel NO demostrados por F1-097. Ramas locales fusionadas glm/157 y codex/verify-pr65 conservadas: sus worktrees son de agentes activos (glm/157 con 4 archivos sucios a las 16:55) y uno está locked; se retiran cuando sus dueños queden inactivos. | main (823a195a3) | crates/telemetry-server/src/profiler.rs; crates/telemetry-server/tests/profiler_loss_contract.rs; docs/PLAN_MAESTRO_SOL_2026-10-18.md; docs/audit/PRESUPUESTO_RECURSOS_2026-10-23.md | 2026-10-23T17:10-05:00 |
| Codex | ACTIVO (ROOT; reanudación 20 Oct) | Cuatro pares privados históricos cerrados; Hidden GREEN 16 numérico conservado y archivo privado sellado; cierre del coordinador INCOMPLETO y Core publicación GREEN 2 no ejecutado. Correcciones públicas sobre main vigente, sin heredar PASS de CONTROL | codex/root-lg5-main-2026-10-15 (e2a2fb72 privado preservado; no publicable por ancestría); siguiente rama pública propia desde main vigente | Plan: docs/audit/REANUDACION_CEROS_REPLAY_2026-10-20.md. F7: scripts y primera CI cloud; ROOT sin Cargo/CI en paralelo. BASE conserva canon histórico de 1863 rutas; recenso 2053/delta 424 preparado, aún no publicado; SI revisión EV RO. Puertos siete, check, regresiones y T1 propios pendientes; meta 100%/72h sin demostración | 2026-10-20 America/Bogota |
| Codex BASE15 | FUSIONADO A MAIN (Ola Ω214 AGY) | PR #65 rescatado e integrado íntegro en main: B15-01 (reduce-only parcial en simulator.rs con contrato 5/5) y B15-02 (validación de parámetros multifractales con contrato 4/4). PR #65 cerrado y rama remota codex/base-audit-2026-10-15 eliminada. Árbol 100% verde y unificado. | main | crates/execution-engine/src/simulator.rs, crates/feature-engine/src/multifractal.rs + tests | 2026-10-22 ~13:17 (rescatado por AGY) |
| Claude (Línea H) | ACTIVO | Ola 3 CERRADA (PR #64 en main 69ccd80e, T-3 v2); ola 4: re-medición T-3 v2 con LH-04 + verificación barata de LH-06 (GLM 122) | claude/linea-h-ola4 (worktree .claude/worktrees/linea-h) | crates/backtest-engine/tests/t3_ventaja_conocida.rs, tests/support/t3_synthetic.rs, docs/audit/linea_h/*, docs/PLAN_LINEA_H_*, buzón/TABLERO/MEMORIA (anexos) | 2026-10-17 ~11:50 (compila sólo por cargo_semaforo.sh; medición T-3 release ≈ 10 min por vehículo) |
| Claude (Línea G) | ACTIVO | Ola 10 ABIERTA: revisión desde la base de los 517 commits entre f39808fdf y e44f57673 (AGY Fase 3/4, Sol F1-093..100, GLM 155-159, Codex QRA, Línea C C-F5/RET), por zonas y con verificación adversarial; después arreglos con test que falla antes. SOLO LECTURA hasta publicar hallazgos (docs/audit/linea_g/ola10_*). Sin Cargo local mientras la RAM libre siga < 2 GB (0,86 GB a las 09:35). | claude/linea-g-ola10 (worktree ../tg-lineag10, fuera del repo) | ninguno en edición (sólo docs/audit/linea_g/ola10_*, plan Línea G §18, esta fila) | 2026-10-24 ~09:40 (Bogotá) |
| Qoder | HUÉRFANO | r7r6 «aprender-medir» — cierre de ronda a medias (rescatado por GLM 132: WIP-commit en su rama) | qoder/r7r6-aprender-medir | buzón, MEMORIA, FORENSIC (rescatados) | 2026-10-10 ~02:06 |
| Fable (Claude Fable 5.1, consejo Quant Sr.) | ACTIVO | EN MAIN hoy: F-A5 (3011c5bbd, LH-04 con el stop real) y F-A6 (24dfc19c8, Hawkes O(1) exacto), ambos con T-1 18/144 y huellas identicas. ATOMO F-A2 EN EDICION AHORA (Fable, no duplicar): gen `funding_rate_sensitivity` (vec[6]) sin lector vivo; su unico lector es `backtest-engine/src/vectorized.rs:208` (evaluador MUERTO, F1-102: sin llamador fuera de sus tests), que escala un funding SINTETICO (clamp(0,5;2)*1e-4) => la evolucion aprende a pagar menos funding ficticio (boton de sobreajuste, familia CSG-14). Arreglo minimo sin cambiar DIMENSION (precedente LG-18/LG-21/G0-4): el gen se congela (sin mutacion, neutro 1,0) y el vectorizado cobra el funding de referencia del exchange (1e-4 por periodo de 8 h, con signo D-671) independiente del genoma; contrato nuevo. Retirar el gen del vector (DIMENSION 144 y genomas persistidos) y cobrar el funding HISTORICO por simbolo quedan como F-A2b para el consejo. T-1 al terminar. LINEA H: repetir T-3 v2 con LH-04 sobre main. | fable/ola1-f2 (worktree .claude/worktrees/fable; docs a main por push HEAD:main) | **EDITANDO**: crates/quantum-arena/src/genome.rs (solo las lineas del gen funding_rate_sensitivity: mutate/random), crates/backtest-engine/src/vectorized.rs (bloque de funding ~195-212), nuevo crates/backtest-engine/tests/fable_funding_gen_congelado_contract.rs | 2026-10-18 ~04:05 |
| Claude (Línea C) | ACTIVO | Censo C-F7b en main: PR #122, #123, #124, #126, #130, #138, #139, #140, #142, #143, #144, #145, #146, #147, #148 y #149 (genes de Kelly de la fila N13, la sombra Kelly y sus revisiones, la fricción al stop de la fila N12, la duodécima tanda, la cuota del exchange para las salidas de la fila N5, las peticiones fuera de la cuota de la fila N20 y la cabecera de los errores firmados de N23 con la espera de exchangeInfo de N33, y la clasificación de los envíos de orden de N25 con el reintento de cuenta de N24; T-1 14/144, como main). Decimoséptima tanda en PR #150 (filas N40, N41 b y N32): CL-203 (un 2xx ilegible de una lectura es un error con su cabecera y se reintenta), CL-204 (el reintento de cuenta sale firmado con la hora de ahora) y CL-205 (oco_probe limpia siempre tras su entrada), y de su revisión CL-206 (refirmar no firma una URL con algo tras la firma) y CL-207 (la limpieza de oco_probe espera al freno 429/418 y lee las órdenes que quedan); filas nuevas N47 a N51. Siguen: N1 y 12b con Sol; N3 con AGY (y el dueño si se persiste); N7 con AGY o el tema 2; N10b con Línea G; N15b, N18, N19, N29, N34, N36, N39, N42 y N44 con Línea I; N16, N30 y N45 con Sol; N17 y N14 con Fable/Línea H; N26 (Línea C con Línea I o AGY); N27 y N28 con Codex QRA/Línea I; N31 god-engine-core; N37 Línea I/AGY; N46 con N50 y N49, N41 a/c, N22, N38 y N43 (Línea C, siguientes); N47 con Sol o los dueños de datos; N48 con Línea I/AGY; N51 con el dueño del forense; f_min_ejecutable de la envolvente y el trinquete del T-1 (dueño); 42 espera a AGY; N8: AGY implementa, Kimi V33 revisa; 36 espera a CazadorConstantes; 28b espera al dueño de Ω121; 4 temas del censo esperan al dueño | claude/auditoria-deslizamiento-apalancamiento-sqtc08 (nube, worktree propio en /tmp) | execution-engine/src/{client.rs (el brazo 2xx de get_payload_with_timeout), executor.rs (refirmar, el reintento de get_payload_account)}; src/bin/oco_probe.rs; execution-engine/tests/n32_oco_probe_source_contract.rs; docs/BINANCE_API_MAP.md §5; docs/audit/linea_g/vetos.tsv (nueve anclas de executor.rs desplazadas); docs/audit/linea_c/censo_motores_2026-10-18.md; buzón | 2026-10-25 ~21:05 (Bogotá) |
| Codex RB (DSH, consejo Quant Sr.; distinto de Codex ROOT/BASE15) | ACTIVO | **RB-H1 EN CURSO (ciclo de vida del proceso en Windows: un motor por entorno, ningún hijo huérfano)**. Evidencia viva en la Demo: (1) `auto_trainer_daemon` HUÉRFANO pid 31716 (padre 31840 muerto, de la corrida a3f 12:33) compite con el del motor actual (pid 20712) por `models/DarkAlpha_BTCUSDT.json` con el MISMO `.tmp`; el job de os-guardian no tiene `KILL_ON_JOB_CLOSE` y el hilo «shutdown-cleanup» es un placebo que duerme; (2) `god_engine.rs` aborta con `expect` en `redb` (`DatabaseAlreadyOpen`, a3g 13:25) DESPUÉS de efectos de arranque; el `state.redb` sólo se escribe (nadie lo lee). Cambio: cerrojo de instancia única por entorno ANTES de cualquier efecto (espera acotada al predecesor), `KILL_ON_JOB_CLOSE`, fuera el diario redb huérfano, entrenador único por símbolo sin `unwrap` al guardar | codex/rb-h1-ciclo-de-vida (worktree C:\Users\jhona\.codex\worktrees\rb-h1-20261110) | crates/os-guardian/src/{lib.rs,ciclo_de_vida.rs}, crates/os-guardian/tests/rb_h1_*.rs, src/bin/god_engine.rs (cabecera de main, bloque redb, supervisor del entrenador, 2 envíos db_tx), src/bin/auto_trainer_daemon.rs, Cargo.toml (redb), Cargo.lock, docs/audit/trinquete_panicos_produccion.tsv | 2026-11-10 21:46 Bogotá |

| Codex F7 (Recursos) | ACTIVO | PR88 reanclada sobre dependencia pública90/7b598f7e y mainfd956f9;29métodos locales GREEN. PróximoCI propio8checks; merge serial90→88→87, sinheredar c813 | codex/f7-recursos-20261020 | tres lanzadores/contratos/workflowF7 y docs propios; no genome_store, Cargo local ni target propio | 2026-10-20 08:25 America/Bogota |

| Claude (Línea I, Opus 5.5) | ACTIVO | I-10 a I-13 en main (PR #167, #168, #169 y #171); I-15 en main (PR #187, fc7f9f8d7: una fila plana de positionRisk no cierra una reserva que espera su llenado) y el arreglo del cuelgue de Sinkhorn que tumbaba el CI (PR #186, 87f00f2a5). I-14 = N44, N34 y N42 en revisión: el vigilante de protección sólo actúa sobre lo que leyó (sin las piernas no coloca nada; con todo cubierto no lee el filtro; la marca de sucio sólo se limpia con positionRisk, el listado y las piernas y el filtro de cada posición leídos; el listado se lee antes que positionRisk; relectura como mucho 5, 10, 20, 40 y 60 s tras acabar una auditoría con un fallo o sin la cuota de lectura, en el ciclo siguiente tras un 429/418, su freno, la cuota crítica agotada o una colocación frenada; la huérfana sin cancelar que sigue abierta pasa a las purgas pendientes; la base de rechazos de la escalada ya no se borra y, en hedge, racha, base y rechazos son por símbolo y lado). Abiertos: N44b, N44c, N34b, N34c, N18b, N15c, N19b, N19c y N48b. | claude/linea-i-i14 (worktree tg-linea-i-i14) | execution-engine/src/vigilante.rs (nuevo), get_symbol_filter de executor.rs (el error de la descarga frenada), quantum-arena/src/protection_health.rs (rechazos por lado), el vigilante de god_engine.rs (ensure_position_protected, purge_orphan_legs, bucle del watchdog), sus contratos y los docs de la línea | 2026-11-08 Bogotá |
*AGY opera por olas intensivas con silencios entre ola y ola: tratar
como activo hasta 48h de silencio.

**Si tu nombre no está: añade tu fila al empezar. Si te vas sin cuota:
pon `SIN CUOTA` en tu fila antes de irte si te es posible.**

**Codex RB, 2026-11-10 — failover de Línea H verificado, nada que rescatar:** el commit
d7d805c89 (rama `claude/linea-h-ola4`, ya retirada) entró en main como 501f05a17
(2026-10-17); sus 16 artefactos (`docs/audit/linea_h/ola4_*`, plan y README)
son idénticos byte a byte en main. No hay worktree `.claude/worktrees/linea-h`.

## 2. Reglas de coordinación

1. **Leer antes de tocar**: verifica en la tabla que ningún agente
   ACTIVO declara los archivos que vas a editar. Conflicto potencial ⇒
   búscalo en el buzón o toma otro frente.
2. **Micro-commits**: pasos pequeños, mensajes autodescriptivos, nada
   a medias sin doc. Ideal: cada paso deja el árbol compilando.
3. **Rama propia → merge → push → borrar rama** (protocolo histórico
   del repo). Verificar `git branch --show-current` antes de CADA
   operación git de escritura (lección GLM 130).
4. **El buzón es append-only**: nunca reescribas entradas ajenas;
   conflictos de merge del buzón se resuelven por UNIÓN (ambos lados).

## 3. Failover (agentes sin cuota)

- Criterio: >48h sin heartbeat NI commits ⇒ HUÉRFANO.
- Procedimiento: (a) auditar su rama/worktree (`status`, `diff`);
  (b) WIP-commit de RESCATE en SU rama (preserva, no mergea a main);
  (c) terminar/mergear a main sólo con contexto suficiente y recibos;
  (d) retirar el worktree para liberar SSD (la rama sobrevive en git);
  (e) actualizar esta tabla + buzón.
- El agente que regresa (horas, días o semanas después): lee ESTE
  archivo + `git log main` ANTES de actuar. Ninguna tarea queda
  abandonada a medias; ningún bloqueo es permanente.

> Secciones 4 (semáforo, sustituido por el mandato 9bdf), 5 (ramas GLM 132), 6 (backlog del consejo)
> y las entradas cerradas están en `AGENT_SYNC_ARCHIVO.md` (integrador, 2026-11-10). Buscar con `rg -n`.

