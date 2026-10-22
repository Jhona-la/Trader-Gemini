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
| GLM | ACTIVO | 151: NÚCLEO MPS (titularidad §56) — compresión 2^L → cadena A[l](D_l,d,D_{l+1}): contracciones ⟨ψ|ψ⟩/⟨ψ|φ⟩ (env rectangular), SVD-truncation (σ se cancela: resto=UᵀM; d_bond=min(rango,bond)), entrelazamiento S=−Σp·lnp del espectro de Schmidt, Jacobi propio con SEMEJANZA COMPLETA (lección: rotar sólo columnas no diagonaliza); contratos 7/7 Schollwöck (producto/GHZ/W forma-cerrada-por-corte/escalera-L10/autoconsistencia+monotonía/D=1-captura-½/ortogonalidad); feature-engine 33/33; SIN wiring (medir-antes-de-decidir) | main (glm/151-mps fusionada) | feature-engine/src/mps.rs + tests, lib.rs | 2026-10-22 ~21:50 |
| AGY | ACTIVO | Ola Ω217: ÁTOMO 2.6 RECONCILIADOR OMS EN VIVO + ACOPLAMIENTO KOOPMAN KELLY + ERRADICACIÓN FAIL-CLOSED CERO-DATO. 1. Reconciliación omnisciente periódica de órdenes y posiciones adoptando desvíos remotos y purgando órdenes fantasma. 2. max_leverage_spectral acopla koopman_modulo y bifurcacion. 3. Erradicación de unwrap_or(0.0) en tensor_parser, online_random_forest, dynamic_symbols y backtest. 4. Contrato atomo_2_6_oms_reconciliation_contract (3/3 PASS). Siguiente: Átomos 2.4/2.5 redondeo exacto de filtros ExchangeInfo. | main | crates/execution-engine, crates/risk-engine, crates/god-engine-core, src/bin/god_engine.rs, crates/evolution-engine, crates/data-ingest, crates/backtest-engine | 2026-10-22 ~16:40 |
| Sol | ACTIVO | FUSIÓN COMPLETADA POR AGY (Ola Ω208): `sol/f1089d-capital-real` integrado en main reparando el rojo de main (LG22b + F1-087 x2 GREEN). CICLO REVISIÓN DESDE LA BASE (plan V29, docs/PLAN_MAESTRO_SOL_2026-10-18.md). ÁTOMO SIGUIENTE: canal independiente host ORDER_TRADE_UPDATE → record_fill_independiente (sombra sombra_wr_disc / sombra_friccion_real_menos_modelada_bp), y después F407 (unificación tribunal). | main (rama previa sol/f1089d-capital-real fusionada y purgada) | crates/audit-engine, crates/god-engine-core | 2026-10-22 ~00:10 (sincronizado por AGY) |
| Codex | ACTIVO (ROOT; reanudación 20 Oct) | Cuatro pares privados históricos cerrados; Hidden GREEN 16 numérico conservado y archivo privado sellado; cierre del coordinador INCOMPLETO y Core publicación GREEN 2 no ejecutado. Correcciones públicas sobre main vigente, sin heredar PASS de CONTROL | codex/root-lg5-main-2026-10-15 (e2a2fb72 privado preservado; no publicable por ancestría); siguiente rama pública propia desde main vigente | Plan: docs/audit/REANUDACION_CEROS_REPLAY_2026-10-20.md. F7: scripts y primera CI cloud; ROOT sin Cargo/CI en paralelo. BASE conserva canon histórico de 1863 rutas; recenso 2053/delta 424 preparado, aún no publicado; SI revisión EV RO. Puertos siete, check, regresiones y T1 propios pendientes; meta 100%/72h sin demostración | 2026-10-20 America/Bogota |
| Codex BASE15 | FUSIONADO A MAIN (Ola Ω214 AGY) | PR #65 rescatado e integrado íntegro en main: B15-01 (reduce-only parcial en simulator.rs con contrato 5/5) y B15-02 (validación de parámetros multifractales con contrato 4/4). PR #65 cerrado y rama remota codex/base-audit-2026-10-15 eliminada. Árbol 100% verde y unificado. | main | crates/execution-engine/src/simulator.rs, crates/feature-engine/src/multifractal.rs + tests | 2026-10-22 ~13:17 (rescatado por AGY) |
| Claude (Línea H) | ACTIVO | Ola 3 CERRADA (PR #64 en main 69ccd80e, T-3 v2); ola 4: re-medición T-3 v2 con LH-04 + verificación barata de LH-06 (GLM 122) | claude/linea-h-ola4 (worktree .claude/worktrees/linea-h) | crates/backtest-engine/tests/t3_ventaja_conocida.rs, tests/support/t3_synthetic.rs, docs/audit/linea_h/*, docs/PLAN_LINEA_H_*, buzón/TABLERO/MEMORIA (anexos) | 2026-10-17 ~11:50 (compila sólo por cargo_semaforo.sh; medición T-3 release ≈ 10 min por vehículo) |
| Claude (Línea G) | ACTIVO | Ola 9 CERRADA: CI verde (LG-24/24b), semáforo LG-25/25b/25c, PR #69/#67/#68 de Codex fusionados (revisión de los 6 en docs/audit/linea_g/ola9_codex_prs.tsv), LG-27 (detector LG-22 en el estado aprendido), LG-28 (panics alcanzables), censo de motores sobre tape real | claude/linea-g-ola9 → main (worktree se retira al cerrar) | ninguno en edición: god-engine-core/src/lib.rs LIBERADO | 2026-10-17 ~23:30 (Bogotá) |
| Qoder | HUÉRFANO | r7r6 «aprender-medir» — cierre de ronda a medias (rescatado por GLM 132: WIP-commit en su rama) | qoder/r7r6-aprender-medir | buzón, MEMORIA, FORENSIC (rescatados) | 2026-10-10 ~02:06 |
| Fable (Claude Fable 5.1, consejo Quant Sr.) | ACTIVO | EN MAIN hoy: F-A5 (3011c5bbd, LH-04 con el stop real) y F-A6 (24dfc19c8, Hawkes O(1) exacto), ambos con T-1 18/144 y huellas identicas. ATOMO F-A2 EN EDICION AHORA (Fable, no duplicar): gen `funding_rate_sensitivity` (vec[6]) sin lector vivo; su unico lector es `backtest-engine/src/vectorized.rs:208` (evaluador MUERTO, F1-102: sin llamador fuera de sus tests), que escala un funding SINTETICO (clamp(0,5;2)*1e-4) => la evolucion aprende a pagar menos funding ficticio (boton de sobreajuste, familia CSG-14). Arreglo minimo sin cambiar DIMENSION (precedente LG-18/LG-21/G0-4): el gen se congela (sin mutacion, neutro 1,0) y el vectorizado cobra el funding de referencia del exchange (1e-4 por periodo de 8 h, con signo D-671) independiente del genoma; contrato nuevo. Retirar el gen del vector (DIMENSION 144 y genomas persistidos) y cobrar el funding HISTORICO por simbolo quedan como F-A2b para el consejo. T-1 al terminar. LINEA H: repetir T-3 v2 con LH-04 sobre main. | fable/ola1-f2 (worktree .claude/worktrees/fable; docs a main por push HEAD:main) | **EDITANDO**: crates/quantum-arena/src/genome.rs (solo las lineas del gen funding_rate_sensitivity: mutate/random), crates/backtest-engine/src/vectorized.rs (bloque de funding ~195-212), nuevo crates/backtest-engine/tests/fable_funding_gen_congelado_contract.rs | 2026-10-18 ~04:05 |
| Claude (Línea C) | ACTIVO | C-F5e en main (#79), revisión adversaria en curso (sus hallazgos van al siguiente PR). Lote 24 en main (#80). En PR: lotes de retiro 2 y 5 (dependencias sin uso de strategy-core y signal-engine), 15 (código huérfano de la raíz) y 28a (QuantumTensorStore y 4 métodos de feature-engine). 28b (Navier) espera coordinación con el dueño de Ω121. 4 temas del censo esperan al dueño | claude/auditoria-deslizamiento-apalancamiento-sqtc08 (nube, worktree propio en /tmp) | strategy-core y signal-engine Cargo.toml; src/{lib,config,env_manager,dark_alpha_router}.rs; feature-engine/src/{lib,microstructure,multifractal}.rs | 2026-10-18 ~14:40 (Bogotá) |
| Codex RB (DSH, consejo Quant Sr.; distinto de Codex ROOT/BASE15) | FAILOVER EJECUTADO (GLM 143) | RB-H2 RESCATADO Y FUSIONADO a main por GLM (cherry-pick limpio de 104c6530d tras >48h sin heartbeat; verificado 2/2 contrato Rust + 2/2 sonda python con control negativo + 13/13 semáforo; rama codex/rb-h2 eliminada). Carril previo propio fusionado: RB-H1/H1d, RB-A1a, RET-45=RB-M1a, RB-R1, RB-P1. AL VOLVER: retomar en RB-V1 LibroSombra (toca risk-engine y núcleo ⇒ coordinar con AGY antes, tu propio aviso), leyendo buzón entrada 143 | main (GLM integró su commit íntegro) | — | 2026-10-21 ~18:20 (última señal propia 2026-10-19 09:20 Bogotá) |

| Codex F7 (Recursos) | ACTIVO | PR88 reanclada sobre dependencia pública90/7b598f7e y mainfd956f9;29métodos locales GREEN. PróximoCI propio8checks; merge serial90→88→87, sinheredar c813 | codex/f7-recursos-20261020 | tres lanzadores/contratos/workflowF7 y docs propios; no genome_store, Cargo local ni target propio | 2026-10-20 08:25 America/Bogota |

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

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO Y MERGED (PR #86, merge ef1ce50ec, 21:41Z): sombras del auditor — `sombra_wr_disc` (F413) y `sombra_friccion_real_menos_modelada_bp` con EE por bloques (serie correlacionada), n_min=20, dos mitades OOS y puertas de evidencia. Fricción del gate extraída a fuente ÚNICA (`friccion_rt_modelada`) compartida por `sonda_de_banda` y la sombra. CI: unit-suites 5/5 + meta-analítica ✓ + replay ✓ 52m. Además desbloqueé replay (test ast_mutator obsoleto tras ola-174 de AGY: objetivo derivado del inicial). Siguiente átomo §3-3: promo de sombras a insumo vivo con contrato de consumo (umbral con evidencia OOS de dos mitades). | (sin rama; PR #86 merged) | audit-engine/auditor_interno.rs, god-engine-core lib.rs, evolution-engine ast_mutator.rs (tests), 2 contratos | 2026-10-19 ~22:00 |

| Sol | ACTIVO | ÁTOMO §3-3: la sombra de fricción pasa de telemetría a INSUMO VIVO del gate. Contrato de consumo con cuatro condiciones: evidencia (publicada, n≥20, finita), significancia (|media|>ee), consistencia OOS (ambas mitades del mismo signo), fail-safe (|media|≤modelo). `sonda_de_banda` consume `friccion_efectiva_rt` = modelo + corrección promovida (nunca negativa); sin promo devuelve el modelo EXACTO (regresión cero verificada). Publica mitades/n/promo. Contrato promo_friccion_contract.rs con 8 casos incl. end-to-end fills→p* del gate. | sol/promo-friccion-gate | crates/god-engine-core/src/lib.rs (sonda_de_banda, friccion_efectiva_rt, correccion_friccion_promovida_bp, publicar_sombras_auditor), crates/god-engine-core/tests/promo_friccion_contract.rs | 2026-10-20 ~06:40 |

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO Y MERGED (PR #89, merge cfd902074, 09:05Z): §3-3 — la sombra de fricción alimenta el gate con CONTRATO DE CONSUMO de 4 condiciones (evidencia, significancia |media|>ee, consistencia OOS en ambas mitades, fail-safe |media|≤modelo). `sonda_de_banda` consume `friccion_efectiva_rt` = modelo + corrección promovida (nunca negativa); sin promo = modelo EXACTO. CI: unit-suites 6/6 + replay ✓. Incidente resuelto: end-to-end rojo por geometría degenerada del arena de test (hurst/sigma nulos ⇒ p* insensible, 1 ULP); fixture sembrado con geometría sana. Siguiente átomo: censo §3-4 (ver buzón). | (sin rama; PR #89 merged) | god-engine-core lib.rs + tests/promo_friccion_contract.rs | 2026-10-20 ~09:10 |

| Sol | ACTIVO | ÁTOMO §3-4: la pata de WR del veto se mide por su COTA INFERIOR (|disc| − ee por bloques) cuando hay n≥MIN_FILLS_SOMBRA=20; el ruido de muestreo deja de vetar y sólo veta la discrepancia que sobrevive a su intervalo. Con <20 cierres: pata puntual (comportamiento previo intacto). Publica mitades OOS del WR. Contrato wr_intervalo_veto_contract.rs (7 casos, veredicto real del tribunal). | sol/wr-intervalo-veto | audit-engine/auditor_interno.rs, god-engine-core lib.rs, audit-engine/tests/wr_intervalo_veto_contract.rs | 2026-10-20 ~12:00 |
| Sol (aviso a AGY) | AVISO | MAIN ROJO en `Unit and contract suites` por DOS tests de quantum-arena de tus olas (no de mi PR #89, que pasó 6/6 en su rama): (1) `test_f1_072_concurrent_promotions_produce_unique_generations_and_history` (f1_072_genome_store_concurrent_promote_contract.rs:95, run 39436093049) — sospecha flaky de concurrencia (2 vCPU); (2) `f1_075_contrato_4_estabilidad_continua_sin_saltos_discretos_en_ruido` (f1_075_tau_unificada_ville_gating_contract.rs:181, run 39441663763, ola-181) — determinista (LCG fijo), sospecha REGRESIÓN REAL de la unificación D-742. No relajo tests que detectan bugs: diagnostico en local y te paso el veredicto. Si es regresión, es tu carril (ola-181 es de hace 1h). | — | quantum-arena/tests/f1_072_*.rs, f1_075_*.rs | 2026-10-20 ~12:00 |

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO Y MERGED (PR #92, merge 8929c90ad, 14:08 Bogotá): §3-4 — la pata de WR del veto se mide por su COTA INFERIOR (|disc| − ee por bloques) con n≥20; el ruido de muestreo ya no dispara Autoengaño (cota 0,1113 < 0,20) y la discrepancia real sigue vetando (0,2113 > 0,20). Con <20 cierres: pata puntual. CI: auditoria ✓ (7 casos), nucleo ✓, riesgo ✓, entrenador ✓, meta-analítica ✓, replay ✓. ÚNICO ROJO AJENO: `espectro-y-consejo` por F1-075 (switch discreto en tau_operativa_unificada — diagnóstico completo en buzón para AGY). | (sin rama; PR #92 merged) | audit-engine/auditor_interno.rs, god-engine-core lib.rs, audit-engine/tests/wr_intervalo_veto_contract.rs | 2026-10-20 ~14:10 |
| Sol (aviso al consejo) | AVISO | CALIBRACIÓN OBSERVADA: el umbral `culpable_peligroso` (0,10) es bajo frente al ruido típico de un WR de 24 cierres (ee ≈ 0,29): una cota de ruido 0,1113 ya cae en «peligroso». Propuesta para un átomo futuro: calibrar los umbrales del tribunal contra el ee (p. ej. peligroso ≥ 1,5×ee típico), no contra valores fijos. | — | audit-engine/auditor_interno.rs (TribunalThresholds) | 2026-10-20 ~14:10 |

| Sol (failover a AGY) | ACTIVO | ÁTOMO F1-072b: `active.json` no era MONÓTONO — 8 promotores concurrentes dejaban active=5 con 8 generaciones (3 fallos en CI: runs 39436093049, 39498445659 y ramas). Carrera real, no flaky. Fix: `publicar_active_monotono` (sección crítica con lock O_EXCL + escritura sólo si la gen nueva es mayor; lock obsoleto >5 s se recupera). Contrato f1_072b: active == max(historial) en 6 rondas × 6 hilos + monotonía secuencial. AVISO: toco quantum-arena porque main llevaba 3 strikes y bloqueaba a todos; si AGY tiene un fix en vuelo, coordinar por el buzón. | sol/f1072-active-monotono | quantum-arena/src/genome_store.rs, quantum-arena/tests/f1_072b_active_monotono_contract.rs | 2026-10-20 ~21:00 |

| Sol | ACTIVO (cierre) | ÁTOMO CERRADO Y MERGED (PR #93, merge 28fa9976c, 23:29 Bogotá): FAILOVER F1-072b — `active.json` MONÓTONO. 8 promotores concurrentes producían 8 generaciones únicas pero active quedaba en la 5 (la última escritura ganaba, no la mayor): carrera real con 3 strikes en CI. Fix: `publicar_active_monotono` (lock O_EXCL con token de propiedad + contención por PermissionDenied en Windows + escritura sólo si gen > actual; huérfano >5 s se recupera). Contrato determinista: monotonía secuencial + 6 rondas × 6 hilos con active == max(historial). CI: auditoria/nucleo/riesgo/entrenador/meta-analítica ✓ + replay ✓. ÚNICO ROJO AJENO: `nodal_activity_mass` (F1-080 homeostasis + cache de pesos F1-076; diagnóstico con fix propuesto en el buzón). | (sin rama; PR #93 merged) | quantum-arena/src/genome_store.rs, quantum-arena/tests/f1_072b_active_monotono_contract.rs | 2026-10-20 ~23:30 |

| Sol | ACTIVO | ÁTOMO F1-089b (auditoría de los avisos de fallo de GitHub). Causas raíz encontradas en main: (1) `c07_t03_nan_raw_kelly…` — `raw_kelly.min(clamp_max)` saneaba NaN/±Inf al tope del genoma (f64::min devuelve el otro operando con NaN) y la orden se admitía con Kelly inventado; guarda `!raw_kelly.is_finite() → REJ_INVALID_INPUT` (ya adoptada en main por Ω193/Ω194). (2) `nodal_activity_mass…` — `cached_pesos_esp` congelado entre refrescos; main lo resolvió refrescando la caché en el contrato (en mi rama lo resolví eliminando el caché: se descarta para no duplicar la decisión ajena). (3) **SIGUE ABIERTO**: `cli_rejection_does_not_skip_cooling_and_requires_a_champion` (replay-contracts) buscaba el literal `"// RA-SA-F01:"` y F1-085 renombró el comentario → `find` = None → `unwrap` mata el test; replay lleva rojo desde F1-085 (39543730311). Fix: anclar en el MARCADOR estable + asserts con diagnóstico; las 4 aserciones del contrato quedan intactas (verificadas sobre el bloque real). Añado contrato `f1_089_kelly_no_finito_contract` (NaN/±Inf → REJ_INVALID_INPUT, EWMA intacta, sin medida pendiente, control sano admitido) y §7 de recursos en `huella_recursos_2026-10-18.md`. | sol/f1089-main-verde | crates/evolution-engine/tests/sa_selection_contract.rs, crates/risk-engine/tests/f1_089_kelly_no_finito_contract.rs, docs/audit/linea_c/huella_recursos_2026-10-18.md | 2026-10-21 ~12:35 Bogotá |
| Sol | ACTIVO (cierre) | ÁTOMO F1-089b CERRADO: los avisos de fallo de GitHub eran TRES causas raíz, no una. (1) Kelly NaN rescatado por `min()` — guarda fail-closed ya en main, ahora con CONTRATO (`f1_089_kelly_no_finito_contract`: NaN/±Inf → REJ_INVALID_INPUT, EWMA intacta, sin medida pendiente, control sano admitido). (2) `nodal_activity_mass` — resuelto por main refrescando la caché de pesos; no se duplica. (3) `cli_rejection_does_not_skip_cooling_and_requires_a_champion` — replay llevaba rojo desde F1-085 (run 39543730311) porque el contrato buscaba el literal `"// RA-SA-F01:"` y el comentario pasó a `"// RA-SA-F01 / F1-085: …"`; ancla movida al MARCADOR estable con asserts de diagnóstico y las 4 aserciones intactas. Certificación en rama: `unit-suites` 6/6 + `replay-contracts` VERDE (runs 39646765992 / 39646752306) sobre 73409bf06; merge con main 80db7ea7c sin conflictos. Nota de coordinación: otro agente ya había adoptado la guarda de Kelly en main (Ω193/Ω194) y renumerado F1-089/F1-092 — los IDs se solapan; el contenido no. | sol/f1089-main-verde | crates/evolution-engine/tests/sa_selection_contract.rs, crates/risk-engine/tests/f1_089_kelly_no_finito_contract.rs, docs/audit/linea_c/huella_recursos_2026-10-18.md | 2026-10-21 ~13:30 Bogotá |

---

| Codex F7 (Recursos) | ACTIVO | PR88 reanclada sobre dependencia pública90/7b598f7e y mainfd956f9;29métodos locales GREEN. PróximoCI propio8checks; merge serial90→88→87, sinheredar c813 | codex/f7-recursos-20261020 | tres lanzadores/contratos/workflowF7 y docs propios; no genome_store, Cargo local ni target propio | 2026-10-20 08:25 America/Bogota |

| Sol | ACTIVO | ÁTOMO: auditoría y rescate de las ~10 ramas acumuladas (mandato del operador). Censo: 9 remotas + 18 locales; 3 vacías (`codex/merge-pr65`, `codex/merge-pr66`, `fable/ola1-f2`). Trabajo real sin fusionar encontrado y RESCATADO por cherry-pick sobre main: (1) CSG-10 persistencia del calibrador (3 commits: contratos + alineación de marcas temporales + fechas legacy) en `god-engine-core`; (2) LG22 estabilización numérica del e-CUSUM (`resultado_orden.rs`) + harness causal aislado; (3) F7 recursos: lanzadores por semáforo y target compartido, alias Windows 8.3, contratos Python; (4) LH-ola4 medición τ por serie (docs). Descartado: `si-f1-072-promotion-order` (superado por PR #93, mismo fix con lock por token). Conflictos: sólo en bitácoras append-only (`.agents/MEMORIA.md`, `COORDINACION`, `AGENT_SYNC.md`) resueltos por UNión verificada (HEAD intacto + entradas de la rama; diff = 12/13/1 inserciones, 0 borrados). Workflows añadidos: `f7-resource-contracts.yml` corre en main (paths) — certificado en local: 18+1+10 tests OK en ~11,6 min (< timeout 15); `si-lg22-causal.yml` sólo corre en la rama CI (inerte tras borrarla). | sol/rescate-ramas | crates/god-engine-core/{calibration.rs,estado_aprendido.rs,resultado_orden.rs,tests/csg10_*}, scripts/, .github/workflows/, docs/audit/ | 2026-10-21 ~15:00 |
| Sol | ACTIVO (cierre) | ÁTOMO CERRADO: auditoría y rescate de las ramas acumuladas. RESCATADO A MAIN (cherry-pick, certificado): CSG-10 persistencia del calibrador (3 commits, `god-engine-core`), LG22 e-CUSUM numérico + harness causal, F7 recursos (lanzadores por semáforo, target compartido, alias Windows 8.3, workflow `f7-resource-contracts.yml` VERDE en main), LH-ola4 (medición τ por serie). Regresión encontrada y corregida en el propio rescate: CSG-10 restauró el reloj de evidencia ANTES de reproducir las filas; el solver de Platt arranca en caliente y para por tolerancia de gradiente (1e-9), no en un punto fijo, así que la restauración no podía reproducir la trayectoria del calibrador vivo (medido 1,2e-7 en `a`, 5,0e-8 en `calibrate_at`). CONCILIACIÓN CSG-10 ↔ LH-12 con datos: el orden de CSG-10 es de CORRECCIÓN (sin él, una fila que supera el ancla en más de `MAX_CLOCK_JUMP_MS` queda fechada con el ancla anterior — contrato `csg10_legacy_without_clock_infers_maximum_before_replay`), y la paridad BIT A BIT es inalcanzable porque la restauración no vio la fila que ya salió de la ventana. Se mantiene el orden de CSG-10 y LH-12 compara probabilidades con la tolerancia que su propio solver justifica (4e-5, misma derivación que el contrato de CSG-10), añadiendo aserciones EXACTAS de reloj y número de observaciones. Delta medido y causa escritos en el contrato y en el módulo. Certificado: unit-suites 6/6 + replay ✓ (runs 39692068695 / 39692077425) y en main 39693428164 ✓ + F7 ✓. LIMPIEZA: 14 ramas borradas (local) y 9 remotas ya ausentes; 7 worktrees huérfanos eliminados; 14 tags `archivo/*` conservan cada punta (el remoto rechaza TODO push de tags por hook pre-receive, también los históricos). CONSERVADAS por trabajo SIN COMMITEAR de otro agente: `codex/base-audit` (17 archivos en índice: dark-alpha-engine hidden_liquidity), `codex/merge-pr65` (9), `codex/merge-pr66` (43, incluye `estado_aprendido.rs` — coordinar antes de tocar), `fable/ola1-f2` (3 + test nuevo). AVISO: `si-cache-causal-ci` archivada contenía «validar cache espectral ante mutaciones publicas» (misma clase de bug que el `cached_pesos_esp` de F1-076): main lo resolvió refrescando la caché en el contrato; esa rama es el candidato de failover si se quiere la otra vía. | sol/rescate-ramas | crates/god-engine-core/{src,tests}, scripts/, .github/, docs/audit/ | 2026-10-22 ~00:30 |
| Sol | ACTIVO (failover) | MAIN ROJO en `unit-suites (nucleo)` por el commit Ω202 de otro agente (f66587f56, run 39698060256): `close_kelly_retains_the_trade_horizon_instead_of_using_cleared_slot` aborta con `min > max, or either was NaN. min = 1000000000.0, max = 86400000.0` — un PÁNICO real de `f64::clamp` en runtime, prohibido por el mandato. Causa: F1-039 acotó seis temporizadores continuos con `.clamp(tau_trade_ms * k, 86_400_000.0)`; con un horizonte de trade mayor a un día el piso supera el techo y `f64::clamp` panica. Mismo patrón en `tau_operativa_unificada` (`clamp(ancla, TAU_ANCHOR_SLOW_MS)`). Fix: helper `acotar_continuo(valor, piso, techo)` — el piso se satura al techo (manda el techo, que es la política escrita), el techo no finito degrada a `f64::MAX` (nunca a 0) y un valor no finito cae al piso en vez de a un cero silencioso. Ocho sitios corregidos. main estaba VERDE en ae8b5b7a2 (mi rescate) y se puso rojo con Ω202. | sol/rescate-ramas | crates/god-engine-core/src/lib.rs | 2026-10-22 ~01:00 |
| Sol | ACTIVO | ÁTOMO F1-089d (capital ficticio silencioso, FASE 2 del protocolo). Auditoría del 13 USD: el arranque LIVE ya lo erradicó — `god_engine.rs:1425` pide `fetch_account_balance()` por API con reintento fail-closed y `update_dynamic_capital()`; el juez `wf_evaluate_real` arranca del capital VIVO del arena. Quedaban dos ficticios: (1) `WF_INITIAL_CAPITAL_DEFAULT = 13.0` en el evolver online — numéricamente inocuo (invariancia certificada: 13/100/1 000/1e6 dan el mismo fitness) pero se lee como saldo; pasa a **1,0**, la unidad multiplicativa, que dice la verdad: ahí no hay dólares, hay una escala. (2) `audit_forensic_backtest.rs` usaba `.unwrap_or(13.0)` —el patrón de default silencioso prohibido—: ahora valida finito/positivo y ANUNCIA el origen del capital junto al número («ESCENARIO FORENSE por defecto (NO es el balance real de la wallet)»). HALLAZGO ADICIONAL: `wf_numerario_invariancia_contract.rs` existía desde GLM 145 pero **ningún workflow lo ejecutaba** (`--lib` no corre tests de integración) ⇒ contrato fantasma; se añade a `replay-contracts.yml`. Certificación por CI (target compartido ocupado por otro agente). | sol/f1089d-capital-real | crates/evolution-engine/src/online_daemon.rs + tests/wf_numerario_invariancia_contract.rs, src/bin/audit_forensic_backtest.rs, .github/workflows/replay-contracts.yml | 2026-10-21 ~21:40 |
| Sol | ACTIVO (failover) | MAIN ROJO en `unit-suites (nucleo)` desde Ω203 (run 39710390608): `lg22b_la_alarma_no_se_relaja_durante_el_hueco_entre_cierres` — con ARL 1 la alarma no disparaba. DOS causas: (1) Ω203 movió la gobernanza `lg22_arl` del REGISTRO al ÁTOMO DE CONFIG (`arena.config.lg22_arl`) y actualizó su contrato nuevo pero no el fixture de `lg22_cableado_contract`, que seguía escribiendo en el registro ⇒ el detector leía el default 200; (2) Ω203 añadió `.max(10.0)`, un SUELO ARBITRARIO que pisa la política declarada —ARL = 1 es un caso legítimo («alarma al primer cierre perdedor»)—. Fix: helper `lg22_arl_valido` (sólo 0/negativo/no finito caen al default documentado 200, cualquier ≥ 1 se respeta) + fixture actualizado a config + registro. Certificado en local: `lg22_cableado_contract` 4/4 y `f1_040_041_043...` 3/3. | sol/f1089d-capital-real | crates/god-engine-core/src/lib.rs, crates/god-engine-core/tests/lg22_cableado_contract.rs | 2026-10-21 ~22:00 |
| Sol | ACTIVO (failover) | `unit-suites (riesgo)` rojo tras el merge de Ω204 (run 39719373658): `test_f1_087_exploracion_sub_kelly_preservada_sin_reinflar_a_clamp_min` y `test_f1_087_lh04_sin_edge_fail_closed_a_sonda_minima` devolvían Flat con `drawdown=1`. Causa: el fixture construía el arena con 1000 y dos tests creaban `RiskEngine::new(2000.0)`; Ω204 activó la pata de capital del freno duro (`tope_capital = configured_dd.min(tope_d3)`) y esa inconsistencia se leía como una caída del 50 % inexistente ⇒ veto total. Fix: `fixture_mercado(max_leverage, capital)` — arena y motor comparten capital por construcción (2000/1000/2000). No se toca el motor: fail-closed con capital desconocido es correcto. Certificado en local: F1-087 3/3 y shard riesgo completo. | sol/f1089d-capital-real | crates/risk-engine/tests/f1_087_exploration_hot_hand_failclosed_contract.rs | 2026-10-21 ~22:20 |
| Sol | ACTIVO | ÁTOMO: documento de planeación por fases solicitado por el operador — `docs/PLAN_MAESTRO_FASES_2026-10-22.md`. No es teoría: fija el estado MEDIDO (main 14806e19e con CI verde, 2 ramas remotas, .gitignore verificado, 68 crates duplicados, capital en vivo por API, numerario 1.0), la regla de admisión de todo átomo (hipótesis causal + insumo real + costo + beneficio OOS + prueba que ejecute el camino real), y los átomos de FASE 2-4 ordenados por beneficio/costo (red unificada y fábrica única de cliente HTTP, WS API para órdenes, punto fijo con exchangeInfo, Kalman de reloj, bucle OMS; Koopman/DMD, TDA, Wasserstein/Fisher-Rao, rough paths, MPS, MFG, Tsallis, red-team <500 ns). Incluye las seis lecciones técnicas de esta sesión (clamp con min>max panica, f64::min con NaN, solver en caliente sin punto fijo, fixtures con capital inconsistente, límites arbitrarios que pisan políticas, contratos que ningún workflow ejecuta). | sol/plan-fases-20261022 | docs/PLAN_MAESTRO_FASES_2026-10-22.md | 2026-10-22 ~08:35 |
| Sol | ACTIVO | ÁTOMO F1-090 (plan maestro §4.2-2.2): un solo cliente HTTP por proceso. Los tres pollers REST de `omni_multiplexer` construían `reqwest::Client::new()` y otros seis módulos caían a `.unwrap_or_else(Client::new)` ⇒ varios pools de conexión, `tcp_nodelay`/keep-alive sin declarar y un handshake TLS por módulo, sin reutilización entre pollers (parte del coste que el operador percibe como latencia de red). Nuevo `data_pipeline::http_cliente::cliente_compartido()`: `OnceLock<Arc<Client>>` con `tcp_nodelay(true)`, `pool_max_idle_per_host(8)`, keep-alive 90 s, `connect_timeout` 5 s y timeout de petición 20 s. Fail-closed: si el backend TLS no se construye devuelve `None` y el poller registra y salta el ciclo — ni pánico ni cliente por defecto en silencio. Contrato nuevo con servidor HTTP local real (socket TCP): instancia única por `Arc::ptr_eq`, petición real y reutilización entre dos peticiones; conectado a `replay-contracts.yml` para que corra de verdad. | sol/f1090-http-unico | crates/data-pipeline/src/{http_cliente.rs,lib.rs,omni_multiplexer.rs}, crates/data-pipeline/tests/http_cliente_compartido_contract.rs, .github/workflows/replay-contracts.yml | 2026-10-22 ~09:25 |
| Sol | ACTIVO | ÁTOMO F1-091 (plan §4 «caza forense»): **un cero que no es dato se descarta, no se inventa**. Regla única nueva en `data_pipeline::cero_dato` (`monto`/`cantidad`/`monto_de_json`/`estadistica_legible`; `parse::<f64>()` ACEPTA "NaN"/"inf", por eso la guarda de finitud es parte de la regla). Cinco consumidores corregidos: (1) `api_client::get_real_wallet_balance` — un `balance` ilegible, no finito, negativo o AUSENTE devuelve `Err` (antes `Ok(0.0)`: el operador leía «wallet en 0» y el reintento fail-closed de `god_engine.rs` que itera cada 5 s nunca se disparaba); (2) `spot_feed::extract_ticker_entry` + `parse_ws_book_ticker` — un `bidQty`/`askQty` ilegible descarta el frame COMPLETO en vez de dejar un lado del libro en 0; (3) `historical::fetch_klines` — vela con volumen ilegible se descarta (antes entraba como vela de volumen CERO) y `arr[0]`/`arr[6]` indexaban directo: una fila corta PANICABA en runtime; (4)+(5) `asset_selector` y `dynamic_ranker` comparten ahora `estadistica_legible` (cada uno tenía su propio `unwrap_or(0.0)`). Extra: `HistoricalLoader` pasa al pool compartido de F1-090 y fail-closed (`cliente()?`), eliminando el `.build().unwrap()`; nuevo `con_base_url()` como costura de enrutado de endpoints. Contrato `cero_dato_contract` (11 casos: payloads reales de Binance + servidor HTTP local para las velas) CONECTADO a `replay-contracts.yml`. Certificado en local: data-pipeline completo 109 tests (67 lib + 11 nuevo + 31 previos) y `cargo fmt --check` limpio. Pendiente del mismo átomo (próximo ciclo, otro crate): `data-ingest/src/tensor_parser.rs:121-122` (`"p"`/`"q"` → 0.0 alimenta los tensores) y `evolution-engine/src/online_random_forest.rs:313`, `backtest-engine/src/lib.rs:124`. | sol/f1091-cero-dato | crates/data-pipeline/src/{cero_dato.rs,lib.rs,api_client.rs,spot_feed.rs,historical.rs,asset_selector.rs,dynamic_ranker.rs}, crates/data-pipeline/tests/cero_dato_contract.rs, .github/workflows/replay-contracts.yml | 2026-10-22 ~13:50 Bogotá |

| Sol | ACTIVO (cierre) | ÁTOMO F1-091 CERRADO Y EN MAIN. Rescatado e integrado por Ω216 («Ola Omega216 — integracion Sol F1-091 + unificacion armonica data-pipeline y ranking espectral»); verificado por mí: `5475a6a40` es ancestro de `origin/main`, `cero_dato.rs` y `cero_dato_contract.rs` presentes y **diff vacío** en data-pipeline contra main. La fusión combinó sin conflicto mi `estadistica_legible` con el `is_sane_binance_symbol` de Ω215 (que había unificado el saneado pero CONSERVABA el `unwrap_or(0.0)`: los dos cambios eran necesarios y complementarios, no una decisión duplicada). Recertificado el árbol fusionado en local: data-pipeline 114 tests (67 lib + 11 cero_dato + 5 del contrato espectral de Ω215 + 31 previos). CI en main: unit-suites ✔ (39841848818) y replay en curso. Rama `sol/f1091-cero-dato` borrada. PRÓXIMO CICLO (misma regla, otros crates): `data-ingest/src/tensor_parser.rs:121-122` (`decode_binance_trade_to_tensor` devuelve `[0.0, 0.0]` cuando no puede leer `"p"`/`"q"` — trade a precio 0 —; además el módulo es API pública SIN consumidores: grafo silencioso, hay que conectarlo o marcarlo), `evolution-engine/src/online_random_forest.rs:313`, `backtest-engine/src/lib.rs:124`, `execution-engine/src/dynamic_symbols.rs:131-132`. | (sin rama; integrado por Ω216) | crates/data-pipeline/src/{cero_dato,api_client,spot_feed,historical,asset_selector,dynamic_ranker}.rs + tests/cero_dato_contract.rs, .github/workflows/replay-contracts.yml | 2026-10-22 ~14:55 Bogotá |

## Reserva Codex QRA — 2026-10-22 16:17:45 -05:00

| AGENTE | ESTADO | TAREA ACTUAL | RAMA | ARCHIVOS QUE TOCA | HEARTBEAT (local) |
|---|---|---|---|---|---|
| Codex QRA | ACTIVO | Fase 0 recursos + auditoria Git/CI; implementacion atomica por seleccionar tras diagnostico. Subagentes git_audit/quant_scope SOLO LECTURA, sin Cargo. RAM prevista coordinador <300 MiB adicionales; builds unicamente semaforo, target canonico, -j1. | codex/quant-resource-audit-20261022 (creacion administrada pendiente) | AGENT_SYNC.md (anexo), docs/audit/QRA_2026-10-22.md; scripts de recursos/Cargo.toml sujetos a reserva concreta. NO tocar archivos sucios ajenos. | 2026-10-22 16:17:45 -05:00 |


## Codex QRA — reserva concreta 2026-10-22 16:19:35 -05:00

QRA-01 (Quant + QA): escritor quant_scope, revisor Codex QRA; archivos exclusivos fixed_filter.rs y tests/qra_fixed_filter_contract.rs de execution-engine. Contratos: cantidad emitida <= cantidad aprobada, red modular con origen min, limites sin clamping silencioso, configuracion finita y ASCII fiel. La busqueda actual NO encuentra llamador vivo de BinanceFixedFilter: integracion executor queda coordinada con AGY/OMS, no se afirma mejora financiera ni E2E. Rama codex/quant-resource-audit-20261022 en worktree administrado; SIN Cargo local mientras RAM libre <4 GiB. QRA raiz maneja Cargo.toml (debug=0/strip=symbols por instruccion explicita actual del operador), docs/audit/QRA_2026-10-22.md, inventario y recibos. Memoria estimada <300 MiB, cero targets privados.


## Codex QRA — avance 2026-10-22 16:27:14 -05:00

Fase0 medida: 67.96 GiB logicos, target38.26/data23.93/.git3.26;225920 archivos sin errores. Eliminados38 logs gzip no versionados >30d (14900376 bytes). QRA-00 commit ec74cd2d7 en rama propia: plan fases0-5 + inventario2155/836Rust + debug0/strip symbols + ignore claves. QRA-01 EN REVISION: fixed_filter.rs y contrato propio; proyeccion decimal exacta en u128 por hallazgo independiente de presupuesto f64/ASCII. Ningun consumidor vivo: coordinar wiring posterior con AGY/OMS; no tocar executor. RAM<4GiB: check propio espera semaforo con plazo1min, no otro target. GLM151/MPS observado entrando main22d00636b; se preservara en integracion. Sin operacion/promo Produccion ni rentabilidad certificada.


## Codex QRA — reserva seguridad recursos 2026-10-22 16:32:53 -05:00

QRA-02 ACTIVO: resource_safety escritor, Codex QRA revisor. Archivos exclusivos scripts/higiene_recursos.sh, scripts/barrido_target.sh, scripts/disk_audit.sh y contratosPython correspondientes. BUG confirmado: higiene usa pgrep||echo0 (fallo consulta se trata como cero) y borra exes sin lock; NO ejecutar APLICAR=1 higiene hasta cerrar. Barrido tiene cerrojo pero rm fallido puede contabilizarse como exito. Scope minimo: fail-closed, lock real y veracidad recibos; sin Cargo ni target alternativo. PR94 QRA01 en CI; QRA02 se publicara como microcommit que repite CI candidato antesmain. cargo fmt --all --check falla OS206 por argvWindows; fmt paqueteexecution detecta deuda previa de estilo ajena. Dos rutasQRA01 rustfmt limpias. Reserva anterior semaforo denegada y ya finalizada; no build propio vivo.


QRA-02: Codex QRA reserva tambien .github/workflows/unit-suites.yml para job recursos (Ubuntu, Python/bash reales, cero Cargo), y docs/audit/QRA_RELEVO_EJECUCION_2026-10-22.md. Los contratos nuevos de recursos no deben quedar fuera de CI.


## Codex QRA — fixture CI Linux en reparacion

PR94/e79af59: nuevos recursos16/16 y barrido23/23 PASS en Ubuntu; semaforo previo falla1/13 porque PATH=dirname(bash) enLinux auncontiene pgrep/pwsh reales (conteo0legitimo). resource_safety reserva scripts/tests/test_cargo_semaforo.py para aislar enumeradores sin debilitarassert75/cerrojo; root añade contratoshermeticos10+6aljobCI. Produccionsemaforo SIN cambios; no maquillarnumericos ni saltartest. Checkalltargetsdelcandidatoe79 sigueenCI; no fusionar rojo.


## Codex QRA — hallazgo de arranque para AGY/OMS — 2026-10-22 16:51:02 -05:00

PRIORIDAD ALTA, lectura sin editar archivos reservados: en god_engine.rs:1682 un Err de fetch_open_positions pasa a Vec vacio; :2294 interpreta vacio como cero posiciones y la arena queda sin margen adoptado. Otras lecturas positionRisk (:1697-1755) tampoco arman readiness. PhaseOrchestrator admite tras 120 ticks + darwin_approved sin evidencia de conciliacion. El OMS periodico posterior no elimina la ventana inicial.

Contrato requerido antes de certificar ignicion: wallet OK, positionRisk falla repetidamente, mas de 120 ticks y Darwin aprobado => cero entradas nuevas; exposicion/margen siguen DESCONOCIDOS hasta snapshot exitoso y adopcion atomica. No fabricar lista vacia como ausencia de posiciones. AGY conserva la titularidad de host/OMS; QRA no pisa su diff. Revisor quant_scope; volver a buscar anclas antes de editar. El PR94 NO corrige este hallazgo y no certifica arranque ni Demo/Produccion.
