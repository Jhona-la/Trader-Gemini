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
| AGY | ACTIVO | Ω148 CERRADA Y EN MAIN (9 hallazgos Línea G Ola 8 cerrados: CSG-2 HIGH, G8-OMEGA-07/08/09/11/15, LG8-145-04/05/06; contratos dark-alpha/risk/signal 100% verdes) + Plan Maestro V19 | main (rama agy/ola-148-helmholtz-riemann-dirac-confluence-hedge-c1 fusionada y retirada) | (ninguno — lock liberado) | 2026-10-17 ~14:16 |
| Sol | ACTIVO | F9-3 CERRADO: isla muerta retirada (−643 líneas, 169/169 verde). Siguiente átomo: skew omni daemon↔vivo (online_daemon.rs:616 vs god-engine-core lib.rs:1980) | sol/f9-muerto-lib51 (mergeado a main; worktree retirado) | evolution-engine/lib.rs (cerrado), online_daemon/god-engine (siguiente) | 2026-10-17 ~14:00 |
| Codex | DORMIDO | Frente B15 (8 ramas: base-audit, root-lg5, si-replay, signal-domain, omega132-expiry, hawkes, quant-reaudit, t3-v2) | codex/*-2026-10-15 | ver ramas | 2026-10-15 ~19:22 |
| Claude (Línea H) | ACTIVO | Ola 3 CERRADA (PR #64 en main 69ccd80e, T-3 v2); ola 4: re-medición T-3 v2 con LH-04 + verificación barata de LH-06 (GLM 122) | claude/linea-h-ola4 (worktree .claude/worktrees/linea-h) | crates/backtest-engine/tests/t3_ventaja_conocida.rs, tests/support/t3_synthetic.rs, docs/audit/linea_h/*, docs/PLAN_LINEA_H_*, buzón/TABLERO/MEMORIA (anexos) | 2026-10-17 ~11:50 (compila sólo por cargo_semaforo.sh; medición T-3 release ≈ 10 min por vehículo) |
| Claude (Línea G) | ACTIVO | Ola 8 CERRADA: LG-23 núcleo hermético, LG-22b, revisión de Ω94–Ω146 (48+1 hallazgos, 6 HIGH), N_eff medido, T-1 18/144 PASA, camino caliente +31 % | claude/linea-g-ola8 → main (worktree se retira al cerrar) | god-engine-core/src/lib.rs, tests lg23/lg22b/lg8_open, 4 bins + 2 vehículos (carga explícita), docs Línea G, buzón | 2026-10-17 ~09:00 (Bogotá) |
| Qoder | HUÉRFANO | r7r6 «aprender-medir» — cierre de ronda a medias (rescatado por GLM 132: WIP-commit en su rama) | qoder/r7r6-aprender-medir | buzón, MEMORIA, FORENSIC (rescatados) | 2026-10-10 ~02:06 |
| Fable (Claude Fable 5.1, consejo Quant Sr.) | SIN CUOTA | EN MAIN: F-A1, F-H1, F-H2, F1 (125 hallazgos, docs/audit/fable). TRASPASO: (1) F-A4 = F1-012/T1-08 commiteado como WIP SIN CERTIFICAR en rama fable/ola1-f2 (drawdown.rs caida_alfa_freno_duro fuera del hilo del evento + contrato fable_freno_duro_cache_contract); correr los tests listados en el mensaje del commit + check all-targets y hacer ff a main. (2) F2: verificacion adversarial de los 18 HIGH en curso al cortarse la cuota; veredictos parciales: F1-005/006/007/008/013 confirmados HIGH por ambos escepticos; F1-001 y F1-011 MED; F1-012 = T1-08 de Linea G. El resultado completo queda en el journal del workflow wf_85e6d27b-632 bajo ~/.claude/projects/C--Users-jhona-Documents-Proyectos-Trader-Gemini/77c95ab7-c9c8-4082-b828-1bb8a264cffa/subagents/workflows/. (3) F1-013 (LH-04: nocional con sl_at_tau genomico, debe usar expected_loss, risk-engine/lib.rs:1491) confirmado HIGH: cambio de tamano vivo, exige aviso a Linea H y T-1. | fable/ola1-f2 (worktree .claude/worktrees/fable, libre para retomar) | ninguno en edicion | 2026-10-17 ~13:51 |

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

- `models/` y `data/` viven SOLO en el checkout raíz (.gitignore):
  las mediciones que necesiten modelos/tapes corren DESDE el raíz
  (lección GLM 131: en worktree la precarga de Ω144 no los ve).
- Build profile: `debug=1` (line-tables-only, decisión Ω134 de AGY:
  reduce .pdb >85%) y `strip=true` en release. Alternativa `debug=0`
  queda registrada para el consejo si la RAM vuelve a apretar.
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

- **Mergeadas ya** (worktree desechable): `claude/linea-g-ola8`.
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
