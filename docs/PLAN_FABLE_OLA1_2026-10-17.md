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

**Mapa (F1-a, cerrado 12:30)**: 22 de 24 crates mapeados por agentes en solo lectura
(`docs/audit/fable/f1_mapa_crates.md`). Cosecha bruta, SIN verificar por humano: 341 símbolos `pub`
sin consumidor fuera de tests (`f1_aristas_muertas.tsv`) y 117 estructuras con crecimiento no acotado
candidatas (`f1_estado_no_acotado.tsv`). Los mapas de `god-engine-core` y de la raíz `src/` cayeron por
cuota y se repiten en la tanda F1-b. Hallazgos destacados del mapa (a verificar en F2):
- `os-guardian`: todo el plano PMU/eBPF/anomaly_detector es inerte por construcción (PMU devuelve
  `None` siempre ⇒ el brazo `(Some, Some)` es inalcanzable); el único canal vivo es
  `panic_memory_dump` (veto de entradas, Relaxed, 5 s de retraso). Contradicción de límites: JobObject
  5 888 MB vs working set recortado al literal 4 096 MB.
- `flight-recorder`, `graph-architecture`, `telemetry-server`: 0 módulos alimentan la decisión viva;
  `telemetry-server` acumula 11 estructuras no acotadas candidatas.
- `dark-alpha-engine`: 6/6 módulos vivos pero 21 símbolos sin consumidor y 7 estructuras no acotadas.

**Hallazgos (F1-b, cerrada 13:25)**: 16 áreas del núcleo de decisión (god-engine-core ×5, risk ×3,
signal ×2, quantum-arena ×2, executor, demonio de evolución, replay, host), una lente combinada, tope
de 8 por área. **125 hallazgos brutos: 18 HIGH, 82 MED, 25 LOW** en `docs/audit/fable/f1_hallazgos.tsv`
(estado «bruto (sin verificar)» hasta F2). Los 24 mapas (ya con `god-engine-core` y la raíz) en
`f1_mapa_crates.md`; 487 aristas muertas y 165 estructuras no acotadas candidatas en sus TSV.
HIGH destacados (sin verificar): F1-011 freno duro «en stops» absorbente sin liberación; F1-012 Monte
Carlo síncrono de 4 M pasos bajo `Mutex` en cada intención con geometría nueva; F1-013 LH-04 convierte
f a nocional con `sl_at_tau` genómico y no con el stop real de la orden; F1-014/015 Cramér-Lundberg
divide R por `max_exchange_leverage` en vez de por la exposición real (×50–320); F1-001 muros L2 falsos
en el modo `trade_only` de la evolución; F1-002/003/004 examen del demonio con recta determinista
hacia el cierre conocido, macro fabricado del retorno previo y rasgos post-resultado en el Shadow
Forest; F1-007 ramas con puerta OBI nunca evaluadas en evolución; F1-008 el Consejo juzga las alts con
la volatilidad de BTC; F1-016/017 sesgo de muestreo λ/μ̂ y `Mutex<VecDeque>` sin cota en el hot path;
F1-018 escalas k≥23 (19,6 h–146 años) pueden dirigir el veredicto espectral.

**Átomo F-A1 (funding per-símbolo, en certificación)** — hallazgo propio durante la lectura del
backlog §6.6: el funding per-símbolo publicado por QO-U1c alimentaba una ARISTA MUERTA
(`fr_elasticity`: se actualizaba y nadie la leía; lo dejaba escrito el propio `train_forest.rs`) tras
un lookup por string en el hot path, y el gen `funding_rate_sensitivity` no tiene lector vivo.
Arreglo: atómico `funding_rate_next` + marca por moneda en `CoinArena` (TTL 360 s, fail-closed a «sin
dato»), publicado por el poller; consumidor vivo = puerta de EV del risk-engine con el coste de carry
esperado E[carry] = d·f·τ/T_F (exacto bajo fase uniforme, causal, sólo coste; T_F = 8 h), también en
`fee_impact`; `FundingRateElasticity` y `update_macro_flow` retirados; `update_macro_features` pierde
el parámetro muerto (un funding NaN ya no veta el OBI de todas las monedas). Contratos:
`quantum-arena/tests/fable_funding_carry_contract.rs` (10) y
`risk-engine/tests/fable_funding_carry_ev_gate_contract.rs` (7). Pendiente: gen muerto
`funding_rate_sensitivity` (átomo F-A2: o se cablea con lift OOS o se retira del genoma) y slot 48 del
tensor 54D (BTC global para todas las monedas; paridad con el entrenamiento de la RED por verificar).

**F2 (verificación adversarial de los 18 HIGH, cerrada 17:40)** — 2 escépticos independientes por hallazgo con
mandato de REFUTAR; 27/36 veredictos (la cuota cortó 9). Tabla: `docs/audit/fable/f2_verificacion.tsv`; la
columna `estado` de `f1_hallazgos.tsv` lleva el veredicto. **CONFIRMADOS HIGH**: F1-005 (política de
rate-limit sólo en rutas de entrada, `executor.rs:1190`), F1-006 (veto del Auditor Interno alimentado por una
`internal_discrepancy` degenerada, `lib.rs:5012`), F1-007 (ramas con puerta OBI nunca evaluadas en la evolución
`trade_only`, `lib.rs:6569`), F1-008 (el Consejo juzga las alts con la volatilidad de BTC, `lib.rs:9023`;
refuta el cierre Ω95/Ω97), F1-013 (LH-04, átomo F-A5). **Rebajados a MED**: F1-001/009/010/011 (el freno
absorbente es decisión del dueño; K a 200 ops vs caída de por vida, documentado en el módulo). **Mixtos**:
F1-002/003/004 (demonio de evolución; necesitan fixture) y F1-012 (= T1-08 de Línea G, cerrado por F-A4).
**Refutado**: F1-014 (duplicado de LC-F4-01). **Sin veredicto por cuota**: F1-015..018. Dueños: AGY tomó
F1-005/006/018 (Ω150); Fable F1-012 (F-A4) y F1-013 (F-A5); F1-007/008 propuestos al dueño de la evolución.

## 6. Átomos priorizados (se completa en F3; los ya identificados van aquí)

| id | sev | qué | archivos | estado |
|---|---|---|---|---|
| F-A1 | HIGH | funding per-símbolo de principio a fin: atómico por moneda → coste de carry en la puerta de EV; arista muerta `fr_elasticity` retirada | quantum-arena/state.rs, risk-engine/lib.rs, god-engine-core/{lib,stateful_engine,math_kernels}.rs, src/bin/god_engine.rs | CERTIFICADO, en main 5895fe178 (17 pruebas) |
| F-A2 | HIGH | gen `funding_rate_sensitivity` (vec[6] de 144): en vivo NO tiene lector; en `backtest-engine/vectorized.rs:208` escala un funding SINTÉTICO (`clamp(0,5;2)·1e-4`), así que la evolución aprende a pagar menos funding ficticio sin efecto real: botón de sobreajuste (CSG-14 de la misma familia). Arreglo: el vectorizado debe cobrar el funding HISTÓRICO real (`{SYM}_FUNDING.parquet`, ya lo descarga `market_context`) y el gen debe retirarse del genoma (cambia DIMENSION=144 ⇒ coordinar con T-1 y con los genomas persistidos) o, mientras, congelarse a 1 | quantum-arena/genome.rs, backtest-engine/vectorized.rs | hallazgo; requiere consejo (DIMENSION) |
| F-A3 | MED | slot 48 del tensor 54D (`combined[48] = omni[11]·1000`): funding de BTC servido a TODAS las monedas para la RED (`swing_nn`); verificar con qué funding se entrenó `DarkAlpha_BTCUSDT.json` y si la red se aplica a altcoins | god-engine-core/lib.rs:2057 | hallazgo |
| F-H1 | MED | política de higiene (logs > 7 d comprimir, graphify dated conservar 3, exes de test > 3 d, worktrees al mergear) como script + regla AGENT_SYNC §4 | scripts/, AGENT_SYNC.md | en main 0d2181347 (`scripts/higiene_recursos.sh`, −1,28 GB aplicados) |
| F-H2 | MED | CI: cada push de SOLO documentación a main lanza «Replay contracts» (≤ 90 min) y «Unit and contract suites» en runners Windows (coste ×2); hoy la mayoría de los pushes son docs (buzón, AGENT_SYNC, TABLERO). `paths-ignore` para `**/*.md`, `docs/**`, `*.tsv`, `graphify-out/**`, `outputs/**` en ambos workflows | .github/workflows/*.yml | en main 4a5476038 (sin protección de rama en main: los pushes directos siguen pasando) |
| F-P1 | HIGH | camino caliente 134 → 175 µs (+31 %): medir con CL-60 tras F-A1 (−1 lookup por string por evento depth), luego atacar las ~90 escrituras/evento de telemetría sin lector (G8-OMEGA-14) y las claves por `&str` | god-engine-core/lib.rs | pendiente (release build, máquina quieta) |
| F-A4 | HIGH | F1-012 / T1-08: el Monte Carlo del umbral del freno duro (20 000 caminos) salía del hilo del evento cada vez que la geometría redondeada (0,02 stops) cambiaba. Caché exacta + vecino CONSERVADOR (mínimo K a radio ≤ 5 pasos) + hilo de fondo que calcula el exacto; mismos umbrales. | risk-engine/src/drawdown.rs, risk-engine/tests/fable_freno_duro_cache_contract.rs (5) | CERTIFICADO, en main 0f12af746 (AGY ola-149 integró el WIP 0b1ffbf80 con radio Manhattan y `max_vecinos`; `cf4_freno_duro` ajustado) |
| F-A5 | HIGH | F1-013 (confirmado por 2 escépticos): LH-04 convertía f a nocional dividiendo por `sl_at_tau(τ)` (curva genómica) y no por `expected_loss` (el stop que la orden lleva, el de CL-2/CL-7/D-744): riesgo realizado f·C·(stop real/stop genómico), con ATR alto varias veces f. Arreglo: divisor = stop real (respaldo a la curva sólo si no hay stop utilizable); telemetría `lh04_f_realizado`, `lh04_capital_asignado`, `lh04_stop_real`; el contrato viejo medía `sl_target` (un PRECIO) como fracción ⇒ cota trivial: endurecido a banda bilateral ±5 % contra el realizado publicado y ≤ f·C. | risk-engine/src/lib.rs (bloque LH-04), risk-engine/tests/lh04_sizing_riesgo_contract.rs, risk-engine/tests/fable_lh04_stop_real_contract.rs (3) | en certificación (pruebas en cola del semáforo; T-1 antes del ff) |

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
- 2026-10-17 12:30 — límite de cuota: F1 cerró con 22/24 mapas y 0/76 auditores; cosecha en
  `docs/audit/fable/`. F-A1 editado (8 archivos + 2 contratos), en `cargo check`. F1-b relanzada
  (16 áreas, 1 lente).
- 2026-10-17 13:25 — F1-b cerrada: 125 hallazgos brutos (18 HIGH) y 24/24 mapas (main 100a0a2a2).
- 2026-10-17 14:10 — F-A1 certificada (17 pruebas) y en main 5895fe178; F-H1 (0d2181347) y F-H2 (4a5476038)
  en main; buzón/MEMORIA/AGENT_SYNC actualizados (c5c3f49bc). Lección: nunca commitear desde la raíz
  compartida (AGY cambió la rama de la raíz a mitad de operación); docs desde el worktree con `push HEAD:main`.
- 2026-10-17 15:40 — F-A4 (F1-012) WIP 0b1ffbf80 publicado SIN certificar; sin cuota: traspaso en AGENT_SYNC.
- 2026-10-17 17:00 — de vuelta. AGY integró y certificó F-A4 en ola-149 (0f12af746; radio Manhattan,
  `max_vecinos`); mi WIP cae en el rebase. F2 cosechada (27/36) y publicada (45247044e).
- 2026-10-17 18:20 — F-A5 (F1-013) editado: divisor = stop real + telemetría; contrato viejo endurecido (medía
  `sl_target`, un PRECIO, como fracción) y contrato nuevo (3). En cola del semáforo: dos cargo AJENOS fuera del
  semáforo (`test -p god-engine-core --lib` desde 16:09 y `check --release --workspace --all-targets --jobs 1`
  desde 17:59), RAM libre 2,7 GB. T-1 (release, -j2, target compartido) antes del ff a main.
