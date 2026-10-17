# PLAN FABLE — Ola F1: revisión desde la base, higiene de recursos y frentes sin dueño (2026-10-17)

> Agente: **Fable** (Claude Fable 5.1, consejo Quant Sr.). Rama `fable/ola1-auditoria-base`,
> worktree `.claude/worktrees/fable`, target compartido (`scripts/cargo_semaforo.sh`).
> Estado operativo en tiempo real: `AGENT_SYNC.md` §1 (fila «Fable»). Narrativa: buzón
> `COORDINACION_CODEX_2026-09-28.md` (append-only). Este documento es el plan de la ola y se
> actualiza en cada micro-commit para que cualquier agente pueda retomarlo si Fable queda sin cuota.

## 0. Protocolo que sigo (AGENT_SYNC §2–§4)

1. Leer `AGENT_SYNC.md` + `git log main` antes de actuar; heartbeat en cada paso.
2. Micro-commits en mi rama; `git add` SOLO de mis archivos (nunca `-A`); árbol compilando.
3. Compilar únicamente con `bash scripts/cargo_semaforo.sh …` (`-j2`, target único, espera si hay
   cargo ajeno o RAM libre < 4 GB). Nunca `cargo test` masivo; `cargo check --all-targets` para tipos.
4. Rama → merge a main (diff contra cada padre + `check --all-targets`) → push → borrar rama + worktree.
5. Anunciar en AGENT_SYNC los archivos de código ANTES de editarlos; evitar los declarados por agentes ACTIVOS.

## 1. Metas del operador (invariantes de diseño)

| Meta | Formalización | Estado medido al llegar |
|---|---|---|
| Crecimiento compuesto +100 % / 72 h desde 13 USD | g = ln 2 / 3 ≈ 0,2310 por día; E[ln(1+f·R)] > g con WR realista 60–80 % | sin evidencia OOS que lo sostenga (T-1 18/144 genes sensibles; T-2 sin re-certificar tras Ω144–Ω146) |
| Inmunidad matemática a la ruina | P(ruina) → 0: Cramér-Lundberg + frenos de drawdown (CL-130…134) + Kelly fraccional con LCB | CSG-3 abierto: LH-04 infla tamaño en rachas (PF literal 5,0, p posterior) |
| Camino caliente < 25 µs/evento | mediana por evento (medidor CL-60) | **≈ 175 µs** (+31 % desde Ω137; LG8-RENDIMIENTO) — puerta no pasa |
| RAM ≤ 16 GB; SSD ≤ 300 GB | semáforo `-j2`; target único; sin worktrees detached | RAM 6,5 GB libres de 23,4; proyecto 76 GB (target 44, data 24,5 intocable) |
| Autoevolutivo / autoadaptable, sin literales arbitrarios | todo parámetro con escritor vivo (registro/genoma/calibración) y lift OOS medido | G8-OMEGA-10: 13 factores multiplicativos, 10 literales; 30 claves de telemetría sin lector |
| Universo multivariante continuo temporal-espectral | 32 escalas diádicas τ_k; ventanas en TIEMPO, no en eventos | G8-OMEGA-17: ventanas de 16/32 EVENTOS en motores físicos (no continuas en τ) |

## 2. Método de la ola (ejecución atómica incremental)

- **F1 — Mapa + hallazgos (en curso, solo lectura)**: 24 mapeadores (uno por crate + raíz) y 76
  auditores (38 áreas × 2 lentes). Lente A: look-ahead/leakage, literales en la decisión viva, aristas
  muertas y nodos silenciosos, errores matemáticos, vetos/límites contradictorios, sesgos discretos,
  overfitting estructural. Lente B: unwrap/expect/panic en producción, NaN/Inf/div0, memoria no
  acotada, hot path (clone/format!/alloc/mutex/strings), concurrencia, resiliencia, persistencia.
  Deduplicación contra `docs/audit/linea_g/ola8_revision.tsv` (48 hallazgos de Línea G).
- **F2 — Verificación adversarial**: cada hallazgo HIGH/MED pasa por 2 escépticos con lentes
  distintas (¿la afirmación es literalmente cierta en el código? ¿es duplicado / qué severidad real
  frente a la meta?). Sobrevive si es real y no duplicado.
- **F3 — Síntesis**: `docs/audit/fable/f1_hallazgos.tsv` (mismo formato que Línea G) + lista de
  átomos (paquetes de trabajo de 1 commit, con test rojo→verde) priorizados por impacto en la meta
  ÷ esfuerzo, con dueño sugerido cuando el archivo lo declara un agente activo.
- **F4 — Ejecución**: un átomo a la vez → test que falla en main → arreglo → `cargo_semaforo.sh
  check --all-targets` → suite del crate → micro-commit → siguiente. Merge a main al cerrar cada
  tanda coherente.

## 3. Estado encontrado (2026-10-17 11:50 local)

- `main` = `origin/main` = 09fe64db1 (Línea H empujó desde el checkout raíz; el raíz es compartido en
  vivo por GLM, Línea G, Línea H, Sol y AGY).
- Ramas sin fusionar: 8 `codex/*-2026-10-15` (Codex DORMIDO desde 2026-10-15; PRs #65–#69 borrador;
  Línea G integra #67 y #69 en `claude/linea-g-ola9`), `claude/linea-h-ola3/4`, `claude/tanda-cf5`
  (rescatada), `qoder/r7r6-aprender-medir` (huérfana, WIP rescatado por GLM 132).
- CI de main: corridas de 69ccd80e (PR #64) en curso; LG-24 declara el primer verde tras Ω144.
- Disco: 76 GB = target 44 (debug/deps 26,5 · incremental 8,8 · release 8,5) + data 24,5 + .git 3 +
  graphify-out 1,6 (cache 0,5 + 12 instantáneas diarias ≈ 0,1 cada una) + logs 0,8 (112 ficheros,
  `demo_v41.log` 107 MB) + .mimosa 0,73 (`hook-state`) + results 0,25. Disco C: 365 GB libres.
- RAM: 6,5 GB libres de 23,4 con 2 cargo + 2 rustc ajenos; ~15 procesos de agentes IA residentes.

## 4. Higiene ejecutada y política propuesta

**Ejecutado (sin tocar nada ajeno activo):**
- `git worktree remove --force .r7r6` + `prune` (Qoder HUÉRFANO > 7 días, árbol limpio, WIP ya en su
  rama). Rama `qoder/r7r6-aprender-medir` intacta (10 commits exclusivos).
- Borrado `NUL.pdb`, `NUL.*.rcgu.o` (artefactos de codegen ignorados, 0,9 MB).

**Pendiente de ventana sin cargo ajeno** (`scripts/disk_audit.sh CLEAN_INCREMENTAL=1`): −8,8 GB.

**Política automática propuesta (átomo H-1, script + regla en AGENT_SYNC §4):**
1. `logs/*.log` > 7 días → comprimir (zstd/gzip) ; > 30 días → borrar. `logs_motor_*.txt` en raíz →
   mover a `logs/`. Flight recorder (`quantum_flight.bin`) con rotación por tamaño.
2. `graphify-out/20*/` no versionadas: conservar las 3 últimas.
3. `target/debug/deps/*.exe` de tests: borrar binarios > 3 días cuando no haya cargo vivo
   (cada binario de prueba ≈ 1 GB con `debug=1`).
4. Worktrees: retirar al mergear; auditoría semanal de `.claude/worktrees` y `~/.codex/worktrees`
   (hoy 13 directorios vacíos externos de Codex: 0 bytes, inofensivos).
5. Telemetría de producción: canal lock-free → hilo secundario → fichero rotado por tamaño (ya hay
   `telemetry-engine`; verificar que ninguna escritura síncrona quede en el hot path — ver F1).

## 5. Hallazgos F1 (se completa al cerrar F1/F2)

Tabla canónica: `docs/audit/fable/f1_hallazgos.tsv`. Resumen por severidad y por crate aquí.

_(pendiente — la auditoría de 100 agentes está en curso; este bloque se rellena en el siguiente
micro-commit)_

## 6. Átomos priorizados (se completa en F3)

_(pendiente)_

## 7. Frentes teóricos que propongo al consejo (backlog §6 sin dueño)

- **Microestructura física en el núcleo del universo espectral** — hoy el core tiene OFI, VPIN,
  Hawkes y profundidad como *features*; falta el modelo de **coste de ejecución** que cierre el lazo
  con el sizing: impacto de Kyle (λ estimado online por regresión recursiva Δp ~ λ·OFI), coste
  temporal/permanente de Almgren-Chriss para dimensionar el tamaño que la liquidez admite,
  posición en cola y probabilidad de fill para maker, latencia medida (reloj local vs `E`/`T` del
  exchange) y comisiones reales por tier. Todo con ventanas en TIEMPO (τ_k), causal, y un contrato
  OOS que demuestre que el coste previsto ≈ coste realizado.
- **Antifragilidad / self-healing en caliente** — cuarentena por motor: si un motor emite NaN/Inf o
  su e-proceso de Ville cae bajo el piso, su voto se congela a abstención informada y se reactiva
  cuando vuelve a pasar la puerta; reconexión WS/API con fail-closed del sizing; detector de cisnes
  (saltos > k·σ espectral) que colapsa el apalancamiento de forma continua, no por escalón.
- **Camino caliente** — medir con CL-60; eliminar las ~90 escrituras/evento de telemetría sin lector
  (G8-OMEGA-14), claves de registro por `&str` → ids enteros precomputados, congelar a 1 los
  multiplicadores sin lift OOS (propuesta Línea G), SIMD en las 32 escalas.

## 8. Bitácora (heartbeat)

- 2026-10-17 11:55 — registro en AGENT_SYNC/buzón (main 6a6ded278); worktree y rama creados;
  F1 lanzada (100 agentes); higiene .r7r6/NUL.*.
