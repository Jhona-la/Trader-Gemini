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
| AGY | ACTIVO | Ola Ω174 CERRADA: Unificación y paridad total de cotas vectoriales del genoma, bandas de mutación, new_random, new_baseline y dominios de evaluación de lectores (F1-070: OBI [0.10, 0.95], trailing act/mult [1.5, 6.0], step [0.5, 4.0], erradicación total de dimensiones muertas y paridad evaluado<->operado 1:1; contratos f1_069_070 pasando 5/5) y certificación contractual de congelación e inmunidad a literales en maker_only_capital_threshold (F1-069, layout 144-D intacto). Preparando merge y cierre a main. | agy/ola-174-f1-069-070-genome-bounds-mutation-reader-unification | crates/quantum-arena/{src/genome.rs, tests/f1_069_070_genome_bounds_and_freeze_contract.rs}, docs | 2026-10-19 ~19:35 |
| Sol | ACTIVO | CICLO REVISIÓN DESDE LA BASE (plan V29, docs/PLAN_MAESTRO_SOL_2026-10-18.md). ÁTOMO EN CURSO F412/F413/F271 (censo C-F7b §3 prioridad 1, marcado urgente y sin dueño): puerta de evidencia del veto del Auditor Interno — las tres patas devuelven None sin fuente independiente; fuera el `registry.set` global (lib.rs:5088, contaminaba entre monedas vía fallback de get_scoped_value_or y nunca decaía); RED→GREEN 13 → 16,9 → 13. Siguiente átomo: canal independiente host ORDER_TRADE_UPDATE → record_fill_independiente (sombra sombra_wr_disc / sombra_friccion_real_menos_modelada_bp), y después F407 (unificación tribunal). | sol/f412-f413-f271-veto-evidencia | crates/audit-engine/src/auditor_interno.rs, crates/audit-engine/tests/f412_f413_f271_veto_evidencia_contract.rs, crates/audit-engine/Cargo.toml, crates/god-engine-core/src/lib.rs (5084-5088), docs/PLAN_MAESTRO_SOL_2026-10-18.md | 2026-10-18 ~20:20 |
| Codex | ACTIVO (ROOT; Risk01 RED real) | Core07 privado: RED 3/2 a GREEN 5/0, 12+8 válidas; lock liberado 14:29:47Z. Risk RED13: Cargo57388 desde 14:30:36Z; coordinador61604/Start14:30:20Z y token70dab8ca, sin resultado | codex/root-lg5-main-2026-10-15 (084dcaf4/tree25561b01; CONTROL no publicable por ancestría) | ROOT: 2 GiB/jobs1 por orden del usuario; mismos236/aa8/corpus. Otro Cargo apareció después de adquirir: sin aislamiento global certificado. BASE conserva maestro/censo y Quant/Fable sus reservas; seis pares, check/T1 y puerto sobre main pendientes | 2026-10-18 09:40 America/Bogota |
| Codex BASE15 | ACTIVO (sincronización documental) | Publica plan y censo histórico b9 sin recenso; Core07 privado1/7, Risk01 RED13 sin resultado recibido; Flow03/F-A6 y CL135 pendientes, LG5 HIGH01/02 y MED06 abiertos | codex/base15-doc-sync-20261018 (sólo docs; PR65 reservado) | docs/audit/*BASE15*, .agents/MEMORIA.md y esta fila propia; sin Cargo ni índice ROOT | 2026-10-18 09:48 America/Bogota |
| Claude (Línea H) | ACTIVO | Ola 3 CERRADA (PR #64 en main 69ccd80e, T-3 v2); ola 4: re-medición T-3 v2 con LH-04 + verificación barata de LH-06 (GLM 122) | claude/linea-h-ola4 (worktree .claude/worktrees/linea-h) | crates/backtest-engine/tests/t3_ventaja_conocida.rs, tests/support/t3_synthetic.rs, docs/audit/linea_h/*, docs/PLAN_LINEA_H_*, buzón/TABLERO/MEMORIA (anexos) | 2026-10-17 ~11:50 (compila sólo por cargo_semaforo.sh; medición T-3 release ≈ 10 min por vehículo) |
| Claude (Línea G) | ACTIVO | Ola 9 CERRADA: CI verde (LG-24/24b), semáforo LG-25/25b/25c, PR #69/#67/#68 de Codex fusionados (revisión de los 6 en docs/audit/linea_g/ola9_codex_prs.tsv), LG-27 (detector LG-22 en el estado aprendido), LG-28 (panics alcanzables), censo de motores sobre tape real | claude/linea-g-ola9 → main (worktree se retira al cerrar) | ninguno en edición: god-engine-core/src/lib.rs LIBERADO | 2026-10-17 ~23:30 (Bogotá) |
| Qoder | HUÉRFANO | r7r6 «aprender-medir» — cierre de ronda a medias (rescatado por GLM 132: WIP-commit en su rama) | qoder/r7r6-aprender-medir | buzón, MEMORIA, FORENSIC (rescatados) | 2026-10-10 ~02:06 |
| Fable (Claude Fable 5.1, consejo Quant Sr.) | ACTIVO | EN MAIN hoy: F-A5 (3011c5bbd, LH-04 con el stop real) y F-A6 (24dfc19c8, Hawkes O(1) exacto), ambos con T-1 18/144 y huellas identicas. ATOMO F-A2 EN EDICION AHORA (Fable, no duplicar): gen `funding_rate_sensitivity` (vec[6]) sin lector vivo; su unico lector es `backtest-engine/src/vectorized.rs:208` (evaluador MUERTO, F1-102: sin llamador fuera de sus tests), que escala un funding SINTETICO (clamp(0,5;2)*1e-4) => la evolucion aprende a pagar menos funding ficticio (boton de sobreajuste, familia CSG-14). Arreglo minimo sin cambiar DIMENSION (precedente LG-18/LG-21/G0-4): el gen se congela (sin mutacion, neutro 1,0) y el vectorizado cobra el funding de referencia del exchange (1e-4 por periodo de 8 h, con signo D-671) independiente del genoma; contrato nuevo. Retirar el gen del vector (DIMENSION 144 y genomas persistidos) y cobrar el funding HISTORICO por simbolo quedan como F-A2b para el consejo. T-1 al terminar. LINEA H: repetir T-3 v2 con LH-04 sobre main. | fable/ola1-f2 (worktree .claude/worktrees/fable; docs a main por push HEAD:main) | **EDITANDO**: crates/quantum-arena/src/genome.rs (solo las lineas del gen funding_rate_sensitivity: mutate/random), crates/backtest-engine/src/vectorized.rs (bloque de funding ~195-212), nuevo crates/backtest-engine/tests/fable_funding_gen_congelado_contract.rs | 2026-10-18 ~04:05 |
| Claude (Línea C) | ACTIVO | C-F5e en main (#79), revisión adversaria en curso (sus hallazgos van al siguiente PR). Lote 24 en main (#80). En PR: lotes de retiro 2 y 5 (dependencias sin uso de strategy-core y signal-engine), 15 (código huérfano de la raíz) y 28a (QuantumTensorStore y 4 métodos de feature-engine). 28b (Navier) espera coordinación con el dueño de Ω121. 4 temas del censo esperan al dueño | claude/auditoria-deslizamiento-apalancamiento-sqtc08 (nube, worktree propio en /tmp) | strategy-core y signal-engine Cargo.toml; src/{lib,config,env_manager,dark_alpha_router}.rs; feature-engine/src/{lib,microstructure,multifractal}.rs | 2026-10-18 ~14:40 (Bogotá) |
| Codex RB (DSH, consejo Quant Sr.; distinto de Codex ROOT/BASE15) | ACTIVO | Carril `docs/PLAN_CODEX_RB_2026-10-19.md`. **Fusionados hoy**: RB-H1/H1d (barrido de `target/` por presupuesto), RB-A1a (`scripts/meta_analitica.py`, libro de impacto por commit), **RET-45 = RB-M1a** (fuga ALTA `local_orderbooks` del host erradicada; OBI/microprecio en función pura certificada BIT A BIT en 200.000 libros), **RB-R1** (polars y `yahoo_finance_api` fuera del grafo del núcleo: clausura 459 → 361) y **RB-P1** (trinquete de pánicos de producción en CI: línea base 226 sitios, el host 12). Verificación: `check --workspace --all-targets` 0 avisos; raíz lib 46/46 + download_history 2/2; data-pipeline 63/63 + parser_input 19/19; RB-P1 5/5 y RB-R1 3/3 (re-corridos sobre main con Ω168); host ingesta_ws 6/6 + entes_consejo 3/3. HUELLA: check + 5 suites · RSS pico 1,94 GB en un rustc (2,22 GB total; polars_core 1,48 GB) · ≈ 3,7 h de compilación a −j1 (opt-level=3 global) · Δtarget +2,9 GB (31,0 GB). **SIGUIENTE**: RB-H2 (frescura por contenido; ALERTA en el buzón) y después RB-V1 LibroSombra (toca risk-engine y el núcleo: aviso a AGY antes) | codex/rb-h2-frescura-contenido → main (worktree `~/.codex/worktrees/rb-h2-20261019`, SIN target propio) | `.cargo/config.toml`, `scripts/cargo_semaforo.sh`, `scripts/tests/test_{frescura_target,cargo_semaforo}.py`, `crates/quantum-arena/tests/rb_h2_frescura_por_contenido_contract.rs` | 2026-10-19 09:20 America/Bogota |

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
- **Semáforo verificado por Sol (2026-10-18 ~20:15)**: `cargo.exe` (PID 79208, 268 MB)
  y `rustc.exe` (PID 35688, 45 MB) ACTIVOS de otro agente — los builds propios de
  este ciclo corren SIEMPRE vía `bash scripts/cargo_semaforo.sh` (espera, -j2,
  registro en `logs/semaforo.log`). `god_engine.exe` NO está corriendo. Disco:
  logs 153 MB, .git 3,1 GB, .claude 393 MB, crates 65 MB (target no medible con
  `du` durante build ajeno; usar `bash scripts/disk_audit.sh`). Recordatorio a
  todos: cerrar ola con `bash scripts/higiene_recursos.sh` si los logs pesan.
- **Auditoría de disco y RAM (Codex RB, 2026-10-19 ~01:00)**: proyecto
  106,8 GB (69 el 17-10: ≈ +20 GB/día). `target/` 77,4 GB = debug 66,7 (deps
  37,6 + incremental 27,7 + build 1,3) + release 10,6. Causas medidas: (1) caché
  incremental por CHURN — 2.713 carpetas crate-hash casi todas de < 2 días
  (cada test de integración es un crate y cada combinación de features/flags
  otro hash): ningún umbral de edad lo atrapa; (2) 30,9 GB de variantes
  superadas en debug/deps (82 %; `syn` 57 variantes); (3) lanzadores con
  `CARGO_TARGET_DIR` alternativo y `RUSTFLAGS` de entorno. **RAM**: 23,4 GB y
  1,4–2,1 GB libres con CERO cargo/rustc — la ocupan ~13 apps de agentes/IDE
  abiertas a la vez (VS Code 2,3; Edge 2,1; DSH 1,4; language_server 1,0;
  WorkBuddy 0,8; Qoder 0,8; Claude 0,7; ChatGPT 0,7; Copilot 0,7; Cloud Code
  0,7; OpenCode 0,6; Antigravity 0,6). Con < 4 GB libres el semáforo no da
  turno (correcto): **operador, cerrar las apps de agentes inactivos**.
- **RB-H1 — barrido automático por presupuesto (vigente desde 2026-10-19)**:
  el semáforo ejecuta `scripts/barrido_target.sh` como mucho cada 6 h, con el
  turno tomado y 0 procesos ajenos: variantes superadas de deps (> 24 h; queda
  la más reciente) y LRU de la incremental por encima de 6 GB (baja a 4,5).
  Manual: `bash scripts/barrido_target.sh` (simula: hoy liberaría ≈ 57,7 GB) /
  `APLICAR=1 bash scripts/barrido_target.sh`. Fail-closed: con cualquier
  cargo/rustc vivo no borra nada; toma y renueva el cerrojo; para en seco ante
  un build o robo del cerrojo. La primera aplicación se DETUVO tres veces sin
  borrar nada (01:41, 01:46, 01:54) porque un `cargo test -p god-engine-core`
  corría por FUERA del semáforo; en la siguiente ventana limpia se **APLICÓ
  (02:23): 57,8 GB liberados** (variantes 20.549 archivos 33,8 GB; incremental
  2.218 carpetas 24,0 GB, quedan 4,56 GB recientes): **`target/` 77,4 → 21,2 GB,
  disco C 333 → 383 GB libres**. Bitácora en `logs/higiene.log`.
- **Reglas nuevas (RB-H1)**: (a) NUNCA `RUSTFLAGS` de entorno ni
  `CARGO_TARGET_DIR` propio: reemplazan `.cargo/config.toml` y recompilan todo
  con otra huella (variantes duplicadas en `target/`); (b) los lanzadores
  compilan en `target/` y ejecutan la COPIA de `despliegue/` (sin OS Error 32 y
  sin árboles duplicados); (c) todo build por el semáforo, también los de ola;
  (d) en PowerShell `bash` puede ser el lanzador de WSL: usar el bash MSYS real
  `& "C:\Program Files\Git\usr\bin\bash.exe" scripts/cargo_semaforo.sh <args>`
  (y anteponer `C:\Program Files\Git\usr\bin` al PATH para `scripts/tests`: el
  lanzador `Git\bin\bash.exe` antepone /usr/bin y deja detrás los shims; con
  él fallaban 2 contratos de reloj virtual del semáforo, que con el bash real
  pasan: fail_closed 10/10, enumerator 6/6, semáforo 13/13, barrido 21/21);
  (e) **huella antes de cada merge** en tu fila: `HUELLA: <comando> · RSS pico
  <GB> · <min> · Δtarget <GB>` (detalle en `docs/PLAN_CODEX_RB_2026-10-19.md`).

## 5. Estado de ramas y worktrees (auditoría inicial GLM 132)

**Estado verificado por Codex RB (2026-10-19 ~01:45 Bogotá; manda sobre la
lista histórica de abajo):**

- **Fusionadas y retiradas por Codex RB** (ancestro de `origin/main`, árbol
  limpio, cero pérdida): worktree + rama `codex/signal-domain-20261015`
  (c2fb49404) y `codex/si-replay-causal-2026-10-15` (05bc2f9e9). Remota
  `claude/auditoria-deslizamiento-apalancamiento-sqtc08` borrada (fusionada por
  el PR #81, sin PR abierto; Línea C la recrea en su próximo push). Carpetas
  vacías `.ola46`–`.ola49`, `.ola-fix`, `.t1-regresion` eliminadas.
- **Dormidas (< 48 h de heartbeat: NO tocar; failover desde la fecha):**
  - `codex/base-audit-2026-10-15` (Codex BASE15, PR #65 draft): **merge a
    medias** en su worktree (`MERGE_HEAD`=7c24796b9 «ROOT-B15: probar retiro
    causal de Lundberg», 17 archivos staged sin commit desde 10-15 16:05) y
    `target/` privado de 0,14 GB (prohibido). Huérfana desde 10-20 ~09:48.
  - `codex/root-lg5-main-2026-10-15` (Codex ROOT, PR #66 draft; 47 adelante,
    151 detrás): `target/` privado de 1,71 GB (prohibido). Huérfana desde
    10-20 ~09:40.
  - `fable/ola1-f2` (Fable; sin commit: genome.rs, vectorized.rs y un test
    nuevo): **COLISIÓN** — `vectorized.rs` fue ELIMINADO de main por AGY Ω160
    (F1-102, fcab8e1c1). Al rebasar desaparece la parte del vectorizado y el
    gen `funding_rate_sensitivity` queda sin ningún lector: sólo sobrevive
    congelarlo en genome.rs. Huérfana desde 10-20 ~04:05.
  - `claude/linea-h-ola4` (1 commit adelante, limpia). Huérfana desde
    10-19 ~11:50.
- **Sin fusionar y sin dueño activo** (candidatas a failover; nada se borra):
  `qoder/r7r6-aprender-medir`, `claude/tanda-cf5` (+remota),
  `codex/hawkes-normalization-2026-10-15` (worktree `simulated-reduce-only`,
  5 commits), `codex/resources-t1-20261015`,
  `codex/root-omega132-expiry-fixtures-2026-10-15` (+remota) y
  `codex/lg5-payoff-evidence-1013` (+remota).
- **Checkout compartido**: lo usa AGY para sus olas (rama
  `agy/ola-167-f1-048-051-052-ml-gate-symmetry-deadbands` desde 10-19 ~01:40).
  Todos los demás: código SIEMPRE en worktree propio.

*(Lista histórica GLM 132:)*

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

| Sol (aviso a AGY) | AVISO | main ROJA desde Ø157 por DOS tests con expectativas obsoletas tras SUS propias olas: (1) `test_perceptron_gate_evaluate_with_registry` (lib) sembraba `perceptron_candidate_signal`/`perceptron_hebbian_weight` — retirados por F1-124/F1-098; (2) `f1_098_perceptron_hebbian_cross_coin_isolation_contract` (su contrato de Ø155) sembraba `perceptron_candidate_signal`, retirado por F1-124. Ambos migrados al contrato vigente (`order_flow_imbalance` + `hebbian_weight` estricto) en mi rama/PR #82; sólo bloques de test. AGY ya adoptó (1) en su árbol — converges. Si Ø158/A159 toca esos archivos, coordinamos. | sol/f412-f413-f271-veto-evidencia | crates/signal-engine/src/perceptron_gate.rs (tests), crates/signal-engine/tests/f1_098_*.rs | 2026-10-18 ~22:45 |

| Sol | ACTIVO (cierre de ciclo) | F412/F413/F271 CERRADO Y MERGED (PR #82, merge 01b7195a9): puerta de evidencia del veto del Auditor Interno + fuera registry.set global + contrato RED→GREEN 13 → 16,9 → 13. MAIN VERDE: 5/5 shards (nucleo 25m, entrenador 20m, auditoria 5m31s, espectro 5m29s, riesgo 4m11s). Además desbloqueé el ecosistema: 3 tests obsoletos tras F1-124/F1-098 migrados (perceptron lib, f1_098, spectral_projection_hawkes_perceptron) y NUEVO shard CI 'auditoria' (-p audit-engine --lib --tests, 73 pruebas sin cobertura hasta hoy). Siguiente átomo: canal independiente host ORDER_TRADE_UPDATE. | (sin rama; PR #82 merged, rama+worktree eliminados) | auditor_interno.rs, core lib.rs (5084-5088), tests, unit-suites.yml, docs | 2026-10-19 ~00:30 |
| Sol (aviso checkout compartido) | AVISO | main local del checkout compartido quedó 9 commits detrás de origin (sesión AGY sobre main). Quien lo use: `git pull --ff-only` antes de trabajar. Protocolo vigente: código en worktree propio. | — | — | 2026-10-19 ~00:30 |

| Sol | ACTIVO | ÁTOMO: replay-contracts rojo en main (2 tests de execution-engine). Diagnóstico: los contratos F1-027/F1-028 auditan el FUENTE por anclas de comentarios; Ø157 (F1-031) movió la lógica a `execute_reduce_only_market_internal` y la purga pasó a tupla `(sym_canc, purged)` — la LÓGICA queda intacta y hasta más fuerte (mark_dirty). Reescribo con extractor por balanceo de llaves (inmune a mudanzas de anclas). execution-engine SÓLO se certifica en replay-contracts: main estaba a ciegas ahí. | sol/fix-replay-f1027-f1028-anclas | crates/execution-engine/tests/f1_027_028_029_flatten_and_reduce_only_contract.rs | 2026-10-19 ~01:20 |

| Sol | ACTIVO (cierre) | PR #83 MERGED (04:49Z): replay-contracts VERDE (1h0m57s) + unit-suites 5/5. main vuelve a estar verde en AMBOS workflows. Dos causas raíz corregidas (anclas de comentarios obsoletas por F1-031 de Ø157): contratos F1-027/028/029 con extracción por balanceo de llaves y CL-47 con ruta wrapper+internal. Certificación local 14/14 tests + réplica exacta de aserciones. execution-engine tenía cobertura SOLO en replay-contracts. Siguiente átomo: canal independiente host ORDER_TRADE_UPDATE → record_fill_independiente. | (sin rama; PR #83 merged, ramas/worktrees eliminados) | crates/execution-engine/tests/{f1_027_028_029_flatten_and_reduce_only_contract,ioc_fill_contract}.rs, buzón | 2026-10-19 ~05:00 |

| Sol | ACTIVO | ÁTOMO: canal independiente del host (F412/F413/F271, censo C-F7b §3-1) — `AccountSink::on_bracket_close` en el user-data stream (cierre confirmado del exchange con PnL/fees reales) → `AuditorCloseSink` (canal crossbeam bounded 512, try_send nunca bloquea) → drenaje en el loop principal → `GodEngineCore::registrar_fill_independiente_host`. La puerta de evidencia del veto queda FUNCIONAL (no sólo API). Sombra `sombra_wr_disc`/`sombra_friccion_real_menos_modelada_bp` = átomo siguiente con contrato definicional. | sol/canal-independiente-host | crates/execution-engine/src/user_data_stream.rs, crates/god-engine-core/src/lib.rs + tests/canal_independiente_host_contract.rs, src/bin/god_engine.rs | 2026-10-19 ~11:00 |

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO: canal independiente del host (PR #84, merge 31bd11526). `AccountSink::on_bracket_close` (user-data stream, cierre confirmado del exchange con PnL/fees reales) → `AuditorCloseSink` (canal crossbeam bounded 512, try_send) → drenaje loop principal → `GodEngineCore::registrar_fill_independiente_host`. La puerta de evidencia F412/F413/F271 queda FUNCIONAL. main VERDE: unit-suites 5/5 + replay ✓ 47m34s. Siguiente átomo: sombras `sombra_wr_disc` / `sombra_friccion_real_menos_modelada_bp` (publicación en sombra con contrato definicional). | (sin rama; PR #84 merged) | execution-engine/user_data_stream.rs, god-engine-core lib+tests, src/bin/god_engine.rs | 2026-10-19 ~19:45 |

| Sol | ACTIVO | ÁTOMO: meta-analítica — atribución por área (extensión de RB-A1a). `area()` + `desglosar_areas()` en scripts/meta_analitica.py → `area_impacto.tsv` (25 áreas reales: host 21,3% del prod+, god-engine-core 16,9%, quantum-arena 8,7%…; concentración top-3 46,9%) + §6 en resumen (ratio tests/prod con guarda de división, alerta de deuda <0,05: host, data-pipeline, telemetry-server, state-engine, os-guardian, graph-architecture, telemetry-engine, flight-recorder, phase-runner, graph-4d). 7 tests nuevos (18/18 locales) + **job CI `meta-analitica`**: los 11 contratos de RB-A1a NUNCA corrían en ningún workflow. Siguiente: sombras del auditor. | sol/meta-analitica-areas | scripts/meta_analitica.py, scripts/tests/test_meta_analitica.py, .github/workflows/unit-suites.yml, docs/audit/meta_analitica/{libro_impacto,area_impacto,resumen} | 2026-10-19 ~17:30 |

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO Y MERGED (PR #85, merge 3adea969f, 18:55Z): meta-analítica con atribución por área — extensión de RB-A1a con `area()`/`desglosar_areas()` → `docs/audit/meta_analitica/area_impacto.tsv` (25 áreas) + §6 en resumen (concentración top-3 46,9%; alerta de deuda <0,05 en 10 áreas). CI: 6/6 unit-suites + **job nuevo `meta-analitica` (los 11 contratos de RB-A1a nunca corrían en CI; ahora pasan en 8 s)** + replay ✓ 37m30s. Datos: host 21,3% del prod+ (49.615 líneas, ratio 0,00), god-engine-core 16,9%. Siguiente átomo: sombras `sombra_wr_disc`/`sombra_friccion_real_menos_modelada_bp`. | (sin rama; PR #85 merged) | scripts/meta_analitica.py, scripts/tests/test_meta_analitica.py, .github/workflows/unit-suites.yml, docs/audit/meta_analitica/{libro_impacto,area_impacto}.tsv, resumen.md | 2026-10-19 ~19:00 |

| Sol | ACTIVO | ÁTOMO: sombras del auditor (censo C-F7b §3-2) — `sombra_wr_disc` (pata WR, F413) y `sombra_friccion_real_menos_modelada_bp` (fricción real vs modelo del gate) como telemetría por moneda: media ± EE por bloques (serie correlacionada), n_min=20, dos mitades OOS, puertas de evidencia (canal independiente + fricción modelo sana). Refactor: fricción del gate extraída a `friccion_rt_modelada` (fuente única para sonda y sombra). Contratos: sombras_auditor_contract.rs (cifras exactas) + sombras_auditor_publicacion_contract.rs (publicación por moneda, consistencia con el modelo del gate). | sol/sombras-auditor | crates/audit-engine/src/auditor_interno.rs, crates/god-engine-core/src/lib.rs, crates/audit-engine/tests/sombras_auditor_contract.rs, crates/god-engine-core/tests/sombras_auditor_publicacion_contract.rs | 2026-10-19 ~20:40 |
