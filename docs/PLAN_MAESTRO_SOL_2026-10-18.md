# PLAN MAESTRO — REVISIÓN DESDE LA BASE (ciclo Sol, 2026-10-18)

> Promovido por Sol a petición del operador: «Inicia una nueva revisión desde la
> base, ya que han cambiado muchas cosas». Convive con
> `docs/PLAN_MAESTRO_SINCRONIZACION.md` (V28 y anteriores, histórico vigente) y
> con `docs/PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md` (archivo por
> archivo). Este documento manda **sólo** donde contradice a los anteriores en
> estado verificado de hoy; donde no, los complementa.
>
> Regla de oro del ciclo: **ejecución atómica incremental** — un módulo, teoría
> o fallo concreto a la vez, implementado a fondo, certificado con pruebas
> reales, integrado a `main`, y sólo entonces el siguiente. Nada a medias.
>
> **Revisión desde la base, 2026-10-23 13:02 Bogotá:** baseline `3b85722e2`;
> estado histórico de las secciones fechadas no equivale al estado actual.
> Coordinar con `docs/audit/QRA_CONTINUACION_PLAN_2026-10-23.md` y el inventario
> por blob ya existente, no duplicar un censo para fingir revisión integral.
> AGY tiene el router de viabilidad/fees; GLM entrelazamiento; Codex controles
> de recursos en el primary sucio. Sol toma sólo telemetría de pérdidas.
>
> Orden operativo renovado: (0) propiedad de archivos/ramas/CI y recursos;
> (1) invariantes de capital, costos, OMS y causalidad antes de cualquier orden;
> (2) trazabilidad de observaciones/métricas y reconexiones por archivo/blob;
> (3) evolución online y persistencia con datasets temporales purgados;
> (4) teoría espectral contrastada por ablación OOS neta y costo computacional;
> (5) Demo con evidencia de fills y circuit breakers, sin promoción de fondos
> automática acreditada por tests offline; (6) observabilidad/visor desacoplado.
> Cada registro exige productor, consumidor, dominio/unidades, prueba ejecutada,
> SHA, período/dataset, costo RAM/latencia y métricas brutas/netas. Inventariado,
> auditado, integrado y económicamente validado son estados diferentes.
>
> **Átomo F1-098:** `profile_node!` usa `let _ = queue.push(...)` y descarta
> muestras cuando el anillo está lleno. Añadir contador saturante de rechazo,
> consumirlo en el snapshot real y log off-thread, probar conservación
> intentos = retenidos + rechazados en un intervalo quiescente, sin cambiar
> sizing/señales/fees ni afirmar mejora de rentabilidad o latencia.
>
> **Corrección de F1-097:** no está demostrado que `FreePhysicalMemory` excluya
> standby; Microsoft lo describe como memoria física no usada y disponible.
> `Page Faults/sec` incluye fallos blandos; su tasa no prueba por sí sola I/O
> al pagefile, fuga del kernel ni causalidad de Defender. El cambio de contador
> no acredita memoria recuperada. No desactivar antivirus, matar procesos
> ajenos o reiniciar sin decisión del operador. RAM disponible medida hoy:
> 1,20 GiB; builds locales sólo con admisión real >=4 GiB y target canónico.
>
> Perfil vigente: `debug=0`, `strip="symbols"` en dev/test según AGENTS; las
> recomendaciones históricas debug=1 de este archivo quedan sustituidas.

---

## 0. La meta y su marco (no se renegocia)

- **Meta financiera**: crecimiento exponencial compuesto, duplicar el capital
  real de la wallet del entorno cada 3 días (≈26 %/día), sin capital fijo ni
  forzar apalancamiento para alcanzar la meta. Se trata como **objetivo a medir**, nunca
  como resultado certificable por matemática avanzada: todo incremento debe
  demostrarse fuera de muestra, tras comisiones, con causalidad estricta.
- **Marco conceptual**: el sistema es un **universo multivariante, continuo,
  temporal y espectral** multiactivo. Scalping/swing y los regímenes discretos
  de volatilidad son **proyecciones** de ese universo, no categorías. Ninguna
  decisión nace de etiquetas de régimen.
- **Criterio de aceptación de cualquier cambio** (del mandato y de AGENTS.md):
  hipótesis, insumo causal, costo/beneficio medible OOS tras comisiones; sin
  valores arbitrarios ni hardcodeados; tests que certifiquen funcionamiento
  real, precisión matemática y conexión total con la arquitectura. Prohibido:
  mocks, stubs, placeholders, TODOs, funciones huérfanas, cálculos que no
  alimenten decisiones reales.

---

## 1. Estado real verificado hoy (2026-10-18 ~20:20, Bogotá)

| Dato | Valor verificado |
|---|---|
| `main` local | `01ce0a57e` (Ø157 de AGY fusionado; GLM 140 sincronizó la divergencia previa main/origin) |
| `origin/main` | igualar main salvo push pendiente de Ø157 — **verificar antes de push propio** |
| Rama de trabajo de Sol | `sol/f412-f413-f271-veto-evidencia` (worktree `.claude/worktrees/sol-atom-f412`) |
| Agentes ACTIVOS (AGENT_SYNC) | AGY (Ø158 en curso), GLM, Codex ROOT/BASE15, Claude Líneas C/G/H, Fable, Sol |
| Procesos de build | `cargo.exe` + `rustc.exe` ajenos activos ⇒ builds propios **sólo** vía `scripts/cargo_semaforo.sh` |
| `god_engine.exe` | no corriendo |
| Disco | proyecto ≪ 300 GB (logs 153 MB, .git 3,1 GB, .claude 393 MB, crates 65 MB) |
| Censo de motores (C-F7b) | 576 símbolos: 363 RETIRAR, 53 ZONA_AJENA, 51 UNIFICAR, 43 CONSERVAR, 28 RECONECTAR_SOMBRA, 20 ENSAMBLE_SOMBRA, 11 YA_RESUELTO, 7 DECISION_DUENO |

**Incidente de concurrencia (lección aplicada)**: un commit de docs de Sol
aterrizó en la punta de `agy/ola-158` por checkout concurrente de AGY en el
árbol compartido. Recuperado por ff-merge en la rama de Sol; la rama de AGY
**no** se tocó (no se destruye trabajo ajeno; `git branch -f` sobre rama
checkouteada por otra sesión está bloqueado por diseño). A partir de este
ciclo, **todo el trabajo de código de Sol corre en worktree aislado**
(`.claude/worktrees/sol-atom-f412`, `CARGO_TARGET_DIR` compartido por el
semáforo) y el árbol compartido se reserva para sincronización puntual.

---

## 2. Diagnóstico de la revisión desde la base

1. **El veto vivo del Auditor Interno estaba roto por tautología** (F412/F413/
   F271, censo §3 prioridad 1): disparaba sin evidencia independiente y no
   decaía. **Átomo en curso en este ciclo** (§4).
2. **El sistema de evidencia del veto depende de un canal que aún no existe**:
   los fills del host (`ORDER_TRADE_UPDATE`) no alimentan ningún auditor. Ese
   canal es la fuente independiente que habilita, en sombra, la reconexión.
3. **~14 % de `src` (≈21 500 líneas) son módulos muertos**; el censo C-F7b da
   veredicto, coste y plan por símbolo. La retirada/reconexión se ejecuta por
   lotes atómicos con anclas rebuscadas antes de cada lote.
4. **Los 4 temas del dueño** (censo §5: telemetría sin lector, ejecución fuera
   de IOC, integridad del veto, teorías sin dato) siguen **bloqueados** a la
   espera de decisión del operador. Este plan no los toca.
5. **Higiene de recursos**: el semáforo (LG-25), la retención ejecutable
   (`higiene_recursos.sh`) y el presupuesto duro (≤16 GB RAM, ≤300 GB SSD, un
   solo `target/`) son política vigente; este ciclo añade la verificación
   explícita de procesos ajenos antes de cada build.

---

## 3. Fases del plan (con dueño sugerido — el dueño real se confirma en AGENT_SYNC)

- **FASE 0 — Sincronización y saneamiento (Sol, este ciclo)**: rama propia,
  heartbeat, plan V29, buzón; verificar main==origin; auditar disco/RAM y
  procesos; aplicar worktree aislado para código.
- **FASE 1 — Integridad de la evidencia (Sol)**: F412/F413/F271 (puerta de
  evidencia) → canal independiente host `ORDER_TRADE_UPDATE` →
  `sombra_wr_disc` / `sombra_friccion_real_menos_modelada_bp` → unificación
  F407 (bloque tribunal: `verify_internal_claims`, `evaluate_tribunal_verdict`,
  `sync_auditor_interno_params`) **después** de F412/F413/F271, como ordena el
  censo.
- **FASE 2 — Motores muertos por censo (todos, por lotes)**: 24 reconexiones en
  sombra por prioridad (censo §3) y 55 lotes RET (§4 del censo), un commit
  atómico por lote, anclas rebuscadas, `Cargo.lock` commiteado.
- **FASE 3 — Teoría nueva con rigor (consejo)**: candidates del censo
  (Garman-Klass σ, Lévy, transfer entropy vs Hawkes, Kalman «precio
  eficiente», Hodge, OBI L2, fricción por latencia-Hurst) entran **primero en
  sombra/offline** con contrato de superioridad OOS; sólo entonces al núcleo.
- **FASE 4 — Microestructura y antifragilidad**: liquidez dinámica, toxic flow,
  slippage/latencia reales, inmunidad a la ruina, self-healing en caliente
  (backlog del consejo, AGENT_SYNC §6, ítems 1-4).
- **FASE 5 — Latencia y recursos**: SIMD/zero-copy en hot path (frente P7),
  telemetría off-thread lock-free, límites duros de RAM/SSD, `debug=1` +
  `strip=true` en release.

---

## 4. Mi carril (Sol) — átomos en orden, con su recibo

1. **F412/F413/F271 (EN CURSO)**: puerta de evidencia + fuera `registry.set`
   global + contrato RED→GREEN 13 → 16,9 → 13. Archivos:
   `audit-engine/src/auditor_interno.rs`, `god-engine-core/src/lib.rs`
   (5084-5088), `audit-engine/tests/f412_f413_f271_veto_evidencia_contract.rs`.
2. **Canal independiente del host** (siguiente): `ORDER_TRADE_UPDATE` →
   `AuditorInterno::record_fill_independiente`; sombra `sombra_wr_disc` y
   `sombra_friccion_real_menos_modelada_bp`; T-1 y golden.
3. **F407 (después del 2, como ordena el censo)**: unificar el veredicto del
   tribunal (hoy sólo en tests) con el veto vivo, o retirar el dúo
   `verify_internal_claims`/`evaluate_tribunal_verdict`; incluye
   `sync_auditor_interno_params` y los átomos `tribunal_*` de config.rs.
4. **RET del censo en mi carril** (backtest-engine y secuencia del plan de
   retiro), un lote por commit.

---

## 5. Protocolo de recursos y concurrencia (vigente para todos)

- Un build a la vez; siempre `bash scripts/cargo_semaforo.sh <args>`
  (deduce `CARGO_TARGET_DIR=<raíz>/target`, `-j2`, bitácora en
  `logs/semaforo.log`). Un cerrojo huérfano (>240 min o sin procesos vivos) se
  libera solo.
- Presupuestos duros: RAM total ≤16 GB; carpeta del proyecto ≤300 GB; prohibido
  duplicar carpetas del proyecto o crear `target/` por rama; worktrees apuntan
  al target compartido; `debug=1` (line-tables-only) y `strip=true` en release.
- Higiene por ola: `bash scripts/higiene_recursos.sh` (simula) /
  `APLICAR=1 ...` (ejecuta); retirar worktrees fusionados
  (`git worktree remove --force` + `prune`); `logs/` rotados y comprimidos.
- Failover: >48 h sin heartbeat ni commits ⇒ HUÉRFANO ⇒ rescate: WIP-commit en
  su rama, terminar, certificar, mergear, retirar worktree, actualizar
  AGENT_SYNC. Ninguna tarea queda abandonada; ningún bloqueo es permanente.
- Commits atómicos y micro; `git status` antes de cada commit; **nunca**
  `git add -A/-u`; sólo archivos propios. Merge a main con verificación
  main==origin y CI verde antes y después.

---

## 6. Bitácora de este ciclo (Sol)

- 2026-10-18 ~20:20 — rama + heartbeat + nota semáforo/disco (954c8cc11).
- 2026-10-18 ~21:00 — átomo F412/F413/F271 implementado
  (`compute_internal_discrepancy_evidenciada`, `record_fill_independiente`,
  publicación condicional en core, `registry.set` global eliminado) con
  contrato `f412_f413_f271_veto_evidencia_contract.rs` (RED→GREEN 13 → 16,9 → 13
  con la función real del Consejo).
