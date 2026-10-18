# AGENT_SYNC.md — Protocolo maestro de coordinación multi-agente

> Creado por GLM (ciclo 132, 2026-10-16) por mandato del operador.
> Regla de oro: **antes de tocar cualquier archivo, lee este documento;
> después de terminar cualquier paso, actualiza tu heartbeat.**
> El canal narrativo sigue siendo `COORDINACION_CODEX_2026-09-28.md`
> (buzón append-only); ESTE archivo es el estado operativo en tiempo
> real: quién está vivo, quién toca qué, y qué recursos se están
> consumiendo.

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
| GLM | ACTIVO | 133: erradicación código simulado (PMU mock→CL-15) | glm/133-erradicar-codigo-muerto | os-guardian/{pmu_sensor,ebpf_core,observability_plane}, buzón | 2026-10-16 ~19:30 |
| AGY | ACTIVO | Ω153 CERRADA: Cierre F1-020 (live_envelope_gate_for_slot por orden y ranura individual en booktick_replay y run_backtest_native con rollback local), F1-021 (cableado de rep.metrics [CVaR95, turnover, Calmar] en SA fitness de evolution.rs), F1-022 (Total Equity en DD/ruina y MTM terminal en last_close_executed sin lookahead), F1-023 (regla causal del tick Lee-Ready en run_backtest_native), F1-122 (hedge_eta default neutral 0.0) y F1-123 (ProyeccionEspectral operabilidad estricta según masa resuelta). 3 nuevos contratos unitarios/integración pasando al 100%. Siguiente: Ola Ω154 (F1-024, F1-025, F1-026 en evolution-engine). | agy/ola-153-f1-020-022-023-122-123 | crates/backtest-engine/src/{booktick_replay.rs, lib.rs}, crates/signal-engine/src/{orchestrator.rs, proyeccion_espectral.rs}, src/bin/evolution.rs, tests | 2026-10-18 ~03:15 |
| Sol | ACTIVO | F9-3 CERRADO: isla muerta retirada (−643 líneas, 169/169 verde). Siguiente átomo: skew omni daemon↔vivo (online_daemon.rs:616 vs god-engine-core lib.rs:1980) | sol/f9-muerto-lib51 (mergeado a main; worktree retirado) | evolution-engine/lib.rs (cerrado), online_daemon/god-engine (siguiente) | 2026-10-17 ~14:00 |
| Codex | ACTIVO (ROOT; Risk01 RED real) | Core07 privado: RED 3/2 a GREEN 5/0, 12+8 válidas; lock liberado 14:29:47Z. Risk RED13: Cargo57388 desde 14:30:36Z; coordinador61604/Start14:30:20Z y token70dab8ca, sin resultado | codex/root-lg5-main-2026-10-15 (084dcaf4/tree25561b01; CONTROL no publicable por ancestría) | ROOT: 2 GiB/jobs1 por orden del usuario; mismos236/aa8/corpus. Otro Cargo apareció después de adquirir: sin aislamiento global certificado. BASE conserva maestro/censo y Quant/Fable sus reservas; seis pares, check/T1 y puerto sobre main pendientes | 2026-10-18 09:40 America/Bogota |
| Claude (Línea H) | ACTIVO | Ola 3 CERRADA (PR #64 en main 69ccd80e, T-3 v2); ola 4: re-medición T-3 v2 con LH-04 + verificación barata de LH-06 (GLM 122) | claude/linea-h-ola4 (worktree .claude/worktrees/linea-h) | crates/backtest-engine/tests/t3_ventaja_conocida.rs, tests/support/t3_synthetic.rs, docs/audit/linea_h/*, docs/PLAN_LINEA_H_*, buzón/TABLERO/MEMORIA (anexos) | 2026-10-17 ~11:50 (compila sólo por cargo_semaforo.sh; medición T-3 release ≈ 10 min por vehículo) |
| Claude (Línea G) | ACTIVO | Ola 9 CERRADA: CI verde (LG-24/24b), semáforo LG-25/25b/25c, PR #69/#67/#68 de Codex fusionados (revisión de los 6 en docs/audit/linea_g/ola9_codex_prs.tsv), LG-27 (detector LG-22 en el estado aprendido), LG-28 (panics alcanzables), censo de motores sobre tape real | claude/linea-g-ola9 → main (worktree se retira al cerrar) | ninguno en edición: god-engine-core/src/lib.rs LIBERADO | 2026-10-17 ~23:30 (Bogotá) |
| Qoder | HUÉRFANO | r7r6 «aprender-medir» — cierre de ronda a medias (rescatado por GLM 132: WIP-commit en su rama) | qoder/r7r6-aprender-medir | buzón, MEMORIA, FORENSIC (rescatados) | 2026-10-10 ~02:06 |
| Fable (Claude Fable 5.1, consejo Quant Sr.) | ACTIVO | EN MAIN hoy: F-A5 (3011c5bbd, LH-04 con el stop real) y F-A6 (24dfc19c8, Hawkes O(1) exacto), ambos con T-1 18/144 y huellas identicas. ATOMO F-A2 EN EDICION AHORA (Fable, no duplicar): gen `funding_rate_sensitivity` (vec[6]) sin lector vivo; su unico lector es `backtest-engine/src/vectorized.rs:208` (evaluador MUERTO, F1-102: sin llamador fuera de sus tests), que escala un funding SINTETICO (clamp(0,5;2)*1e-4) => la evolucion aprende a pagar menos funding ficticio (boton de sobreajuste, familia CSG-14). Arreglo minimo sin cambiar DIMENSION (precedente LG-18/LG-21/G0-4): el gen se congela (sin mutacion, neutro 1,0) y el vectorizado cobra el funding de referencia del exchange (1e-4 por periodo de 8 h, con signo D-671) independiente del genoma; contrato nuevo. Retirar el gen del vector (DIMENSION 144 y genomas persistidos) y cobrar el funding HISTORICO por simbolo quedan como F-A2b para el consejo. T-1 al terminar. LINEA H: repetir T-3 v2 con LH-04 sobre main. | fable/ola1-f2 (worktree .claude/worktrees/fable; docs a main por push HEAD:main) | **EDITANDO**: crates/quantum-arena/src/genome.rs (solo las lineas del gen funding_rate_sensitivity: mutate/random), crates/backtest-engine/src/vectorized.rs (bloque de funding ~195-212), nuevo crates/backtest-engine/tests/fable_funding_gen_congelado_contract.rs | 2026-10-18 ~04:05 |
| Claude (Línea C) | ACTIVO | C-F7a (#75) en main. C-F5b: la prueba de reversión de StatArb contrasta también a la escala de decisión τ_k (EST-3; EST-2 por la intersección), en PR. C-F7b: censo de motores desconectados (421 símbolos, sólo lectura) en curso | claude/auditoria-deslizamiento-apalancamiento-sqtc08 (nube, worktree propio en /tmp) | **EDITANDO (C-F5b)**: strategy-core/src/{prueba_reversion,stat_arb}.rs y sus tests; god-engine-core/src/lib.rs sólo un comentario (~:6476) | 2026-10-18 ~09:50 (Bogotá) |

*AGY opera por olas intensivas con silencios entre ola y ola: tratar
como activo hasta 48h de silencio.

**Si tu nombre no está: añade tu fila al empezar. Si te vas sin cuota:
pon `SIN CUOTA` en tu fila antes de irte si te es posible.**

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

## 4. Semáforo de recursos (presupuestos duros del operador)

| Recurso | Presupuesto | Uso (auditoría GLM 132, 2026-10-16) |
|---|---|---|
| RAM total | ≤ 16 GB durante builds | 8,4 GB libres de 23,4 — **regla `-j2`** en horas de ola ajena (`CARGO_BUILD_JOBS=2`; ya default de `scripts/oracle_witness.sh`) |
| Carpeta del proyecto | ≤ 300 GB SSD | ~85 GB (target 34 + data 24 + .claude 13 + resto) ✓ |
| `data/` (tapes) | INTOCABLE | 24 GB, activo del operador |
| target/ | ÚNICO compartido | prohibido crear targets por worktree: usar `CARGO_TARGET_DIR=<raíz>/target`; `target/debug/incremental` se limpia ENTRE olas (`scripts/disk_audit.sh CLEAN_INCREMENTAL=1`, guard: 0 builds activos) |
| Compilación/medición pesada | registrar antes de lanzar | los vehículos `#[ignore]` (700–900 s) y builds release completos cuentan: anota en tu fila «corriendo X hasta ~hh:mm» |
| **Semáforo (LG-25, obligatorio)** | un build a la vez | compila SIEMPRE con `bash scripts/cargo_semaforo.sh <args de cargo>`: fuerza el target compartido y `-j2`, espera si hay otro cargo/rustc vivo o la RAM libre < 4 GB, y registra en `logs/semaforo.log`. Un cerrojo huérfano (agente sin cuota) se libera solo |


**Adaptador ROOT para validación protegida:** V2/V3 PowerShell revisados operan el mismo `target/.semaforo.lock` canónico, con adquisición atómica, cero Cargo/rustc, PID/StartTime, renovación propia y drenaje natural del Job antes de liberar. Por orden directa del operador del 18Oct («No, toca que trabajes así, están libres como 3 gigas»), ROOT usa la copia V3 low-memory con admisión física ≥2 GiB y un solo job; el V3 original mantiene 4 GiB. La excepción afecta sólo la admisión de recursos de ROOT, con RAM real medida antes y después de adquirir; no altera las guardas de fuente/parámetros/resultados ni el default de otros agentes. Es una excepción de lanzador para ejecutar runner236/wrapperaa8 sin modificar sus bytes ni12+8; los demás builds mantienen el comando bash. Es más restrictiva que la paciencia LG25b: nunca omite la observación de procesos. No concede ventana ni acuerdo de Fable por esta nota; ROOT respeta su T1 y sólo arranca con coordinación y lock reales.

- `models/` y `data/` viven SOLO en el checkout raíz (.gitignore):
  las mediciones que necesiten modelos/tapes corren DESDE el raíz
  (lección GLM 131: en worktree la precarga de Ω144 no los ve).
- Build profile: `debug=1` (line-tables-only, decisión Ω134 de AGY:
  reduce .pdb >85%) y `strip=true` en release. Alternativa `debug=0`
  queda registrada para el consejo si la RAM vuelve a apretar.
- **Auditoría de memoria y disco (Claude Línea G, 2026-10-17 ~10:30)**:
  carpeta del proyecto 69 GB (target 37 = debug 29 + release 8,3; data 24;
  .git 3,1; graphify-out 1,6; logs 0,8) — muy por debajo de 300 GB.
  `~/.codex/worktrees` 0,8 GB (el de base-audit lleva un `target/` privado
  de 143 MB: prohibido, a retirar por Codex). RAM: 23,4 GB totales, 7,1
  libres; las apps de agentes ocupan ≈ 5,5 GB (Claude 1,8; ChatGPT 1,0;
  ZCode 0,8; WorkBuddy 0,7; Qoder, Codex y Antigravity 0,4 cada una). El
  riesgo real son las compilaciones simultáneas (0,4–0,9 GB por rustc):
  de ahí el semáforo. `graphify-out/20*/` (instantáneas diarias de hasta
  26 MB) ya no entra en git (.gitignore, LG-25).
- Higiene por ola: `bash scripts/disk_audit.sh` al cerrar; retirar
  worktrees mergeados (`git worktree remove` + prune); no duplicar
  carpetas del proyecto.
- **Política de retención ejecutable (F-H1, Fable 2026-10-17)**: `bash
  scripts/higiene_recursos.sh` (simula) / `APLICAR=1 bash scripts/higiene_recursos.sh`
  (ejecuta) al cerrar cada ola: logs > 7 d gzip, .gz > 30 d borrar, instantáneas
  `graphify-out/20*` no versionadas conservar 3, exes de test > 3 d sólo sin cargo vivo,
  `worktree prune`. Bitácora en `logs/higiene.log`. Nunca toca data/, models/, ni worktrees.
- **CI (F-H2)**: los pushes de solo documentación (*.md, docs/, *.tsv, graphify-out/,
  outputs/, .agents/) ya no lanzan `Replay contracts` ni `Unit and contract suites`.
  Heartbeats y buzón son gratis en runners; el código sigue pasando por CI.

## 5. Estado de ramas y worktrees (auditoría inicial GLM 132)

- **Mergeadas ya** (worktree desechable): `claude/linea-g-ola8` (retirada
  por Claude Línea G); `claude/auditoria-deslizamiento-apalancamiento-sqtc08`
  (PR #70, borrada en remoto el 2026-10-17).
- **Dormidas ~1 día** (NO tocar): 8 ramas `codex/*-2026-10-15` (B15),
  ~~`claude/linea-h-ola3`~~ (CERRADA 17-10 por el PR #64: los 181 archivos eran un merge de `main` sin concluir que Codex dejó en el worktree; nada propio se perdió).
- **Huérfanas rescatadas por GLM 132**: `qoder/r7r6-aprender-medir`
  (WIP-commit de sus 5 archivos), `claude/tanda-cf5` (rescate del test
  `zz_medir_duracion_sonda.rs`; worktree de 12,6 GB retirado).
- **Activas**: `sol/f9-oos-selection-2026-10-16`.
- Externos `~/.codex/worktrees`: 9 directorios vacíos eliminados por
  GLM 132 (0 bytes, sin pérdida); los 5 vivos se respetan.

## 6. Backlog del consejo (del mandato del operador — sin dueño)

1. **Rigor anti-leakage** en motores reconectados: causalidad estricta
   en t, sin look-ahead (la familia GLM 123-128 es el patrón), OOS
   demostrable por motor.
2. **Microestructura física**: liquidez dinámica, toxic flow, slippage
   real, latencia de ejecución en el core del universo espectral.
3. **Antifragilidad/self-healing**: aislamiento y reparación de módulos
   defectuosos en caliente sin parar la operativa; cisnes negros;
   caídas de conexión.
4. **Ultra-baja latencia**: SIMD/zero-copy en el hot path (frente P7
   del mapa C-F3 con propuesta P5 ya escrita), erradicar `.clone()`/
   format! por evento.
5. Calibración `avg_conviction` con hablantes (propuesta GLM 129,
   ×19/14 — requiere oráculo + re-baseline).
6. ~~cablear `Arc<OmniState>` al loop de evolución~~ **CERRADO por Sol
   F9-3 (2026-10-17)**: la isla muerta (`start_evolution_loop`+
   `frozen_macro`) fue RETIRADA, no cableada — el loop vivo no necesita
   los literales y god-engine ya sirve macro con productores reales.
   **SIGUE ABIERTO**: funding per-símbolo (el poller premiumIndex es
   BTC-only; god lib.rs:2763/:4016).
7. **perf_event_open real** para el PMU sensor (GLM 133 dejó
   `Option<PmuVector>` — cuando exista el backend nativo, `Some`
   vuelve a alimentar el detector de anomalías OOB).
