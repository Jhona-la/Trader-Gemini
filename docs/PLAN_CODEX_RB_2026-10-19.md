# Carril Codex RB — revisión desde la base (2026-10-19)

> Agente: **Codex RB (DSH, consejo Quant Sr.)**, distinto de las sesiones
> Codex ROOT y Codex BASE15. Rama de trabajo efímera `codex/rb-*` en worktree
> propio bajo `~/.codex/worktrees/`, **sin `target/` propio** (todo build por
> `scripts/cargo_semaforo.sh`).
>
> Relación con los demás planes: complementa el plan vigente V29
> (`docs/PLAN_MAESTRO_SOL_2026-10-18.md`, fases 0–5) y el censo de motores
> (`docs/audit/linea_c/censo_motores_2026-10-18.md`). **Dueños y archivos en
> edición: manda `AGENT_SYNC.md`.** Este documento no crea un plan maestro
> nuevo (ya hay ~15): fija el carril de Codex RB dentro del V29.

## 1. Método (orden del operador, aplicado a cada átomo)

1. **Meta** → **concepto** → **matemática/estadística/física** → **código**;
   de la arquitectura base al detalle; de lo más importante a lo menos.
2. Cada átomo declara: hipótesis, insumo causal (sólo información disponible
   en t), contrato RED→GREEN con la función real, integración de punta a
   punta con el núcleo (prohibidos mocks, stubs, placeholders, TODOs,
   funciones huérfanas y cálculos que no alimenten decisiones reales),
   `cargo check --workspace --all-targets` limpio y **huella de recursos**
   registrada antes del merge (§4).
3. Ejecución atómica incremental: un átomo a la vez, a `main`, y el siguiente.

## 2. Átomos del carril, en orden

| # | Átomo | Fase V29 | Estado |
|---|---|---|---|
| RB-H1 | Higiene de `target/` por **presupuesto** + lanzadores con un solo target | 5 | **Entregado** (§3); RB-H1d: deps sólo por encima de 40 GB |
| RB-M1 | Fugas de memoria en runtime del motor vivo | 5 | Censo hecho (§6); **RB-M1a = lote RET-45** entregado (§7); M1b/M1c «sin acción» alineados con C-F6; M1d → RB-A1c |
| RB-P1 | Trinquete de pánicos (`unwrap`/`expect`/`panic!`/`assert!` en producción) | 5 | **Entregado** (§7): contrato en CI + línea base versionada |
| RB-R1 | **polars fuera del grafo del núcleo** (recursos): `polars` entra en `god-engine-core`/`backtest-engine` SÓLO por `data-pipeline` (`macro_data.rs`, `market_context.rs`, que sólo usa la herramienta `src/bin/download_history.rs`); `evolution-engine` lo declara sin usarlo. Mudar los dos módulos a la herramienta y quitar la dependencia muerta: los shards de CI más pesados y los builds diarios de todos dejan de compilar la pila polars (`polars_core` medido a 1,2 GB de RSS en un solo rustc) | 5 | **Entregado** (§7): MacroFetcher retirado, fetcher de funding mudado a su herramienta; clausura de `god-engine-core` 459 → 361 paquetes; contrato anti-regresión sobre `Cargo.lock` |
| RB-H2 | **Frescura por contenido en el `target/` compartido**: dos worktrees producen los mismos nombres de artefacto (hash de la ruta relativa al workspace) y cargo decide la frescura por mtime ⇒ un agente puede ejecutar el código de otro worktree sin saberlo (sonda mínima: por mtime el árbol 2 imprime «ARBOL-1»; con `-Zchecksum-freshness`, «ARBOL-2»). `[unstable] checksum-freshness = true` en `.cargo/config.toml` + contrato | 5 | **Listo en su rama** (sonda E2E 2/2 con control negativo; se fusiona justo después de este merge; coste único: recompilar los crates del workspace una vez) |
| RB-A1 | **Meta-analítica del impacto histórico de cada cambio** (mandato del operador): libro por commit de `main` con átomo, agente y métricas de cómputo (clausura de dependencias, coste de build del semáforo, tamaño por crate), de salud del código (pánicos de producción, tests) y financieras (T-1, replays) con deltas por átomo | 5 | **RB-A1a entregado** (dac445628: `scripts/meta_analitica.py`, 11 contratos sobre repos git reales; libro de 1.076 commits en `docs/audit/meta_analitica/`); A1b/A1c en §9 |
| RB-R2 | **polars fuera del paquete raíz**: el paquete raíz (lib `quantum_engine` + 40 bins) depende de polars sólo por 4 herramientas (`download_history`, `feature_validator`, `multi_coin_simulator`, `parquet_to_bin`); cualquier `cargo test -p trader-gemini-v5` compila la pila polars (medido el 19-10: 99 min a −j1 con opt-level=3). Mudarlas a un crate de herramientas del workspace | 5 | Pendiente (después de RB-V1; tocar sólo bins sin dueño activo) |
| RB-V1 | **LibroSombra** (censo §3 prioridad 2: F250/F272/F400, sin dueño): qué vetos destruyen valor | 2 | **RB-V1a entregado** (§11): consejo y ML-GATE en sombra. **RB-V1b entregado 2026-11-10** (§12): 8 puertas del riesgo posteriores a la geometría en sombra, más un veredicto siempre válido (Ville) que corrige la parada opcional de RB-V1a. Siguiente: veredictos vivos de la Demo |
| RB-S1 | Antifragilidad: cuarentena en caliente por motor votante | 4 | Pendiente (coordinar con dueños del orquestador) |
| RB-N1 | **El reloj del host vuelve a vivir**: la guardia HOST-016 cancelaba el sincronizador NTP/Kalman antes de su primera muestra (off-by-one, desde 97996eba0 del 2026-09-17) ⇒ la deriva del reloj entraba entera en la edad del dato que bloquea entradas | 2 | **Entregado 2026-11-10** (§10) |
| RB-N2 | **Puerta de latencia con fundamento**: el umbral de pánico es un gen sin señal en backtest (semilla `1000·π`, hoy 6 945 ms); Kalman de 2 estados (offset + deriva) con R adaptativa y puerta de RTT relativa al mínimo observado | 2 | Propuesto (§10.5); coordinar con Línea I (I-2b/I-5) |
| RB-C1 | **La CI no esconde rojos**: `--no-fail-fast` en las 25 invocaciones de `cargo test` de unit-suites y replay-contracts (el shard `nucleo` se cortaba en `f69_002` y ocultaba el resto) | 1 | **Entregado 2026-11-10** (contrato `rb_c1_ci_sin_corte_temprano_contract`) |
| RB-G1 | **Censo de alcanzabilidad desde el host**: qué módulos no tocan el flujo real (62 sólo los usa el grafo vivo Ω220, que el host nunca evalúa; 116 sin usuario de producción) — de censo léxico a contrato | 1 | Censo léxico publicado (§10.6, `docs/audit/rb_g1_censo_alcance_host_2026-11-10.txt`) |

**RB-M1 — fugas de memoria en runtime.** Concepto: el motor corre 24/7; toda
colección que crece por evento sin cota es una fuga aunque Rust no «pierda»
memoria. Método: censo de `Vec`/`VecDeque`/`HashMap`/`String` alimentados en el
hot path de `god-engine-core`, `quantum-arena`, `signal-engine` y del host
(`execution-engine`, `god_engine.rs`) sin evicción ni capacidad fija; claves
del registro por símbolo o por evento que nunca se retiran; `format!`/`clone`
por evento. Contrato por arreglo: tras N eventos sintéticos el tamaño queda
acotado (O(1) en N) y la salida del motor es bit a bit la misma (T-1 cero).

**RB-V1 — LibroSombra.** Concepto: un veto es una predicción («esta intención
pierde»); su valor se mide contrafactualmente. Matemática (del censo): H0
«el veto acierta» (E[R neto] ≤ 0), error estándar por bloques de τ, Bonferroni
sobre ~10 fuentes, n_min = (z·ŝ/δ)² con δ = fricción/SL. Sombra pura: no
cambia ninguna decisión (pureza bit a bit), indexada por
`veto_registry`/`REJECT_NAMES` (F267) sin desplazar los índices 7, 9 y 14
(F251). Respuesta directa al mandato «auditar que los vetos, bloqueos,
límites y rechazos tengan sentido».

**RB-S1 — cuarentena por motor.** Concepto: un votante que emite NaN/Inf o
entra en pánico no debe contaminar el consenso ni detener la operativa.
Disyuntor por motor (cerrado → abierto con voto 0 → semiabierto con backoff
exponencial), telemetría por motor y prueba de inyección de fallas.

**RB-P1 — trinquete de pánicos.** Línea base medida por crate de
`unwrap`/`expect`/`panic!` fuera de `#[cfg(test)]`; un contrato impide que el
número suba y cada ola lo baja.

## 3. RB-H1 — entregado el 2026-10-19

**Medición (checkout raíz, ~01:00):** proyecto 106,8 GB (69 GB el 17-10;
≤ 300 GB pero creciendo ~20 GB/día). `target/` 77,4 GB = debug 66,7
(deps 37,6 + incremental 27,7 + build 1,3) + release 10,6; `data/` 23,9;
`.git` 3,1; `~/.codex/worktrees` 2,6.

**Causas medidas:**

1. **Caché incremental por churn**: 2.713 carpetas crate-hash, casi todas de
   < 2 días. Cada test de integración es un crate con su caché y cada
   combinación de features/flags de cada agente es otro hash: el disco se
   llena **sin envejecer**, así que ningún umbral de edad lo atrapa.
2. **Variantes superadas en `deps/`**: 30,9 GB de 37,6 (82 %) no son la
   variante más reciente de su artefacto (`syn` 57 variantes, `windows_*` 67).
   El mtime de `deps` dice cuándo se compiló, no cuándo se usó: barrer por edad
   borraría `rlib` vigentes.
3. **Lanzadores**: fijaban `CARGO_TARGET_DIR=target_persistent/
   target_release_god/target_prod` (árboles release completos duplicados) y
   `RUSTFLAGS` de entorno, que **reemplaza** los `rustflags` de
   `.cargo/config.toml` y obliga a recompilar todo con otra huella.
   `LAUNCHER.bat` ejecutaba un binario de una carpeta que ya no existía.

**Entregado:** `scripts/barrido_target.sh` (variantes superadas > 24 h +
LRU de la incremental por presupuesto 6 GB → 4,5 GB; fail-closed con builds
vivos o cuenta desconocida; toma y renueva el cerrojo del semáforo; para en
seco ante un build o robo del cerrojo; sólo dentro de `target/`), barrido
automático en `cargo_semaforo.sh` cada 6 h con el turno tomado, paso 3b en
`higiene_recursos.sh`, cinco lanzadores con un solo `target/`, sin `RUSTFLAGS`
y con copia de ejecución en `despliegue/` (fin del OS Error 32 sin duplicar
árboles; mismo nombre ⇒ dos motores no arrancan a la vez), kill-switch sin
`taskkill` global de `python.exe`. Contratos: `scripts/tests/
test_barrido_target.py` 21/21 y `test_cargo_semaforo.py` 13/13.

**RAM:** 23,4 GB totales y 1,4–2,1 GB libres con **cero** cargo/rustc: la
consumen las apps de agentes/IDE abiertas a la vez (VS Code 2,3 GB, Edge 2,1,
DeepSeek Harness 1,4, language_server 1,0, WorkBuddy 0,8, Qoder 0,8,
Claude 0,7, ChatGPT 0,7, Copilot 0,7, Defender 0,7, Cloud Code 0,7,
OpenCode 0,6, Antigravity 0,6). Con < 4 GB libres el semáforo no concede turno
(correcto). Recomendación al operador: cerrar las apps de agentes inactivos.

## 4. Huella de recursos antes de cada merge (propuesta para todos)

Una línea en la fila propia de `AGENT_SYNC.md` y en el mensaje del commit de
cierre: `HUELLA: <comando> · RSS pico <GB> · <min> · Δtarget <GB>`. RB-H1:
`HUELLA: tests de scripts (python+bash) · <0,1 GB · 3 min · Δtarget 0
(sin compilar Rust)`.

## 5. Hallazgos abiertos para sus dueños (no tocados por Codex RB)

- **Builds fuera del semáforo**: el 2026-10-19 a las 01:38 y 01:45 un
  `cargo test -p god-engine-core` (contrato `f1_048_051_052`) corrió por fuera
  del semáforo y detuvo dos veces el barrido (correctamente: sin borrar nada).
  Regla vigente: todo build por el semáforo.
- **Contratos del semáforo sensibles al lanzador de bash (diagnosticado)**:
  `test_cargo_semaforo_fail_closed::
  test_successful_tasklist_numeric_count_retains_lg25b_patience` y
  `test_cargo_semaforo_enumerator_exit_status::
  test_pgrep_success_known_two_retains_numeric_patience` fallan SÓLO si `bash`
  es el lanzador `C:\Program Files\Git\bin\bash.exe`: antepone
  `/mingw64/bin:/usr/bin` al PATH, el shim de `date` (reloj virtual) queda
  detrás del `date` real y la paciencia nunca llega. Con el bash MSYS real
  (`Git\usr\bin\bash.exe`) pasan las cuatro suites: fail_closed 10/10,
  enumerator 6/6, semáforo 13/13, barrido 21/21. No es un defecto del
  semáforo. La CI (`windows-2022`) no ejecuta `scripts/tests`: propuesta de job
  barato con el bash real.
- **`opt-level=3` global**: `.cargo/config.toml` lo aplica a TODOS los
  perfiles (dev/test incluidos): cada rustc de debug compila en O3, con más
  RAM y tiempo. Medir RSS pico y duración de `cargo test -p god-engine-core`
  en O3 frente al perfil antes de decidir (consejo).
- **`bash` en PowerShell** puede resolver al lanzador de WSL (WindowsApps), que
  sin WSL no arranca: usar el bash MSYS real `"C:\Program Files\Git\usr\bin\bash.exe"`.
- **RB-P1, línea base heurística**: ≈149 sitios (102 `unwrap`, 42 `expect`,
  5 `panic!`) en `src/` fuera de `mod tests`; sobrecuenta funciones `#[test]`
  de nivel superior y módulos gateados por `cfg(test)` desde el padre (p. ej.
  los 11 `expect` de `temporal_spectrum.rs` son todos de tests). El trinquete
  necesita ese filtro antes de fijar la cifra.
- **Buzón de 12.666 líneas** que cada agente vuelve a leer: propuesta de
  archivo mensual en `docs/archivo/` con índice (requiere acuerdo: los
  appends concurrentes chocan).
- **Barrido aplicado (2026-10-19 02:23)**: tras tres detenciones limpias por
  builds fuera del semáforo, 57,8 GB liberados — `target/` 77,4 → 21,2 GB,
  disco C 333 → 383 GB libres.

## 6. RB-M1 — censo de fugas de memoria en runtime (2026-10-19, sólo lectura)

Escenario: 13 activos × ~10 eventos/s ≈ 11 M eventos/día. Bucle caliente:
`process_event` (core `lib.rs:2488`) → `process_tick_dual` (`:3787`). Host:
`src/bin/god_engine.rs` (no `crates/execution-engine/src/bin`).

| Sev | Dónde | Qué crece | Por qué no hay cota | Arreglo (átomo) |
|---|---|---|---|---|
| ALTA | `src/bin/god_engine.rs:3349-3352`, `:3562-3565`; `src/orderbook.rs:29-30` | `local_orderbooks[*].bids/asks` (BTreeMap) | cada `@depth5` inserta el nivel 0; un snapshot nunca trae qty 0, así que `remove` jamás corre; sólo `clear()` al reconectar (≤ 24 h). **Nadie lo lee** | **RB-M1a**: retirarlo (conducta bit a bit idéntica; quita además un O(log n) por mensaje) |
| MEDIA | `crates/execution-engine/src/order_registry.rs:235` | `orders` (HashMap) | `prune_terminated` sólo purga terminales; las que quedan en New/PartiallyFilled (timeouts, -2013, terminales perdidos al reconectar) no salen nunca; `cleanup_stale_orders` no tiene llamadores | **RB-M1b**: cablear `cleanup_stale_orders` + `resolve_via_rest` al bucle de 60 s, con caducidad para `New` con `order_id` 0 |
| MEDIA | `crates/execution-engine/src/trade_accounting.rs:362-412` | `JOURNAL_CACHE` + `position_journal.jsonl` | la compactación sólo corre al arrancar; cada recarga relee el archivo entero | **RB-M1c**: mapa incremental `(símbolo, lado)` o compactar cada 60 s |
| BAJA | core `lib.rs:4002`; `symbol_registry.rs:213`; risk `evidence.rs:196`/`sonda.rs:117`; `consejo_seniors.rs` | asignaciones por evento (String de símbolo, `to_uppercase`, `format!` de claves, copia de 512 ticks) | no crecen, pero cuestan en la ruta caliente | **RB-M1d** (latencia): símbolo internado, comparación sin asignar, buffers en pila reutilizados |
| BAJA | `god_engine.rs:3224` (`tx_estado`), `trade_accounting.rs:229` (`FILLS_LINE_TX`) | canales `mpsc` sin cota | sólo crecen si el escritor de disco se cuelga | canal de capacidad 1 / acotado con contador de descartes |
| ACOTADA | omniscient-registry (~262 nombres × moneda; meseta 5–10 k claves), calibradores (512/200), `spectral_tape` pending (~900), consejo (≤1000), lead_lag, demonio (400/1000/500/5000), freno (4096), colas (1024/5000) | — | ventanas o claves fijas | publicar `registry.len()` para vigilar la meseta |

**Cruce con el censo C-F6 de Línea C** (`docs/audit/linea_c/estado_sin_cota_2026-10-18.tsv`,
539 estructuras, 2 escépticos por candidato, fusionado en el PR #74): las dos
MEDIA de arriba están allí CONFIRMADAS con impacto «real/low» y decisión «sin
acción» (sólo huérfanos ambiguos de ≈ 0,5 KB; una tupla por entrada
confirmada), y los logs y avisos sin tope ya los cerraron CL-139…CL-146. Para
no duplicar ni contradecir, **RB-M1b y RB-M1c quedan cerrados como «sin
acción» (alineados con C-F6)**; RB-M1d (asignaciones por evento) pasa a la
atribución del +31 % del camino caliente (RB-A1c), que es donde pesa.

**Para medir fugas en vivo hay que mirar Private Bytes, no Working Set**: el
host compacta el working set bajo presión (`god_engine.rs:2311-2323`). Los
núcleos de examen del demonio (≈25 por ronda cada 3 min, ~39 MB cada uno) son
churn de asignación, no fuga.

## 7. Entregas del 2026-10-19 (madrugada): RET-45, RB-R1, RB-P1

**RET-45 = RB-M1a (lote 45 del censo, «raíz host»).**
- F472 UNIFICAR — fuga ALTA erradicada: `local_orderbooks` (dos `BTreeMap` por
  símbolo alimentados con el nivel 0 de cada `@depth5`; un snapshot nunca trae
  cantidad 0, así que nada se borraba hasta la reconexión; nadie lo leía).
  `orderbook::microestructura_mejor_nivel` (pura, sin estado ni asignaciones)
  calcula OBI y divergencia del microprecio del snapshot. Certificación: paridad
  BIT A BIT con la fórmula previa del host en 200.000 libros aleatorios
  (precios 1e-8…1e6, spreads 1e-6…1e-2, cantidades 0…1e7) ⇒ T-1 cero por
  construcción; identidad cerrada `div = obi·(p_a − p_b)/(p_a + p_b)` (el
  microprecio de Stoikov se aparta del medio medio spread por el OBI);
  antisimetría; neutral y finita ante entradas no finitas; contrato de conexión
  (el host la usa, sin libros por símbolo, sin copia en línea de la fórmula).
- F480: parser L1 retirado (su contrato migra al nivel 0 de depth5, n = 1).
- F481/R176: daemon duplicado retirado. R30: `fred_daily` retirado, doc del
  módulo y banner corregidos, `unwrap` de `create_dir_all` manejado.

**RB-R1 — la pila polars fuera del grafo del núcleo.** Ver la fila de la tabla
§2. Medido en `Cargo.lock`: clausura de `god-engine-core` 459 → 361 paquetes,
`backtest-engine` 463 → 365, `evolution-engine` 477 → 379; sale
`yahoo_finance_api` y 12 dependencias exclusivas suyas (13 paquetes del lock).
Cero cambio de conducta. Contrato `rb_r1_grafo_del_nucleo_sin_polars_contract.rs`.

**RB-P1 — trinquete de pánicos.** `crates/quantum-arena/tests/
rb_p1_trinquete_panicos_contract.rs` (shard 3 de CI). Recorre el árbol de
módulos real (raíces lib/main/bin y rutas del manifiesto; `mod x;`, módulos en
línea, `#[path]`), excluye `#[cfg(test)]`, `#[test]`, `#[tokio::test]`,
`#![cfg(test)]`, comentarios y literales, y cuenta por paquete `unwrap`
(+`unwrap_err`), `expect` (+`expect_err`), pánicos (`panic!`, `unreachable!`,
`todo!`, `unimplemented!`) y asertos (`assert!`, `assert_eq!`, `assert_ne!`).
Ningún paquete puede subir respecto a `docs/audit/trinquete_panicos_produccion.tsv`;
al bajar, `TRINQUETE_ESCRIBIR=1` regenera la línea base (el historial git del TSV
es la serie histórica de la quema). Línea base del 19-10: 344 archivos de producción, 226 sitios (unwrap 138, expect 75, pánico 8, aserto 5); los del proceso vivo primero: host `god_engine.rs` 12, god-engine-core 7, quantum-arena 7 (`state.rs` 5), execution-engine 4, feature-engine 4, risk-engine 2. Contratos del lexer y del árbol de módulos
con solución conocida. **Procedimiento de merge:** si `main` avanzó, regenerar la
línea base sobre el árbol fusionado ANTES del push.
## 8. RB-V1 — LibroSombra conectado (diseño ejecutable)

**Meta → concepto.** El crecimiento compuesto depende tanto de lo que el sistema
deja de hacer como de lo que hace: un veto que bloquea operaciones con
esperanza neta positiva es una fuga de crecimiento invisible (nunca se ve el
PnL de lo no operado). Un veto es una predicción («esta intención pierde») y se
juzga contrafactualmente.

**Lo que ya existe en `main`.** `audit_engine::shadow_ledger::LibroSombra`
(QS-R4a: memoria fija de 512 intenciones, primer toque TP/SL/vencimiento,
R neto de fricción, veredicto `AhorraDinero`/`BloqueaGanadoras`/`SinEvidencia`
con z·SE; contrato `qs_r4_libro_sombra_contract.rs`) y
`risk_engine::ultimo_rechazo()` (QS-R4b: razón `REJ_*` thread-local de la última
evaluación). `god-engine-core` ya depende de `audit-engine` (usa
`AuditorInterno`): no hay ciclo. **Falta conectarlo** (censo: RECONECTAR_SOMBRA).

**Matemática.**
- Por fuente f: rᵢ = (+tp | −sl | retorno al vencer − fricción)/sl; H0_f «el veto
  acierta»: E[r] ≤ 0. `BloqueaGanadoras` si m − z·SE > 0.
- Dependencia: intenciones solapadas de la misma (fuente, moneda, lado) no son
  independientes; el libro admite UNA abierta por clave (ya implementado) y el
  SE se calcula sobre resoluciones, con bloques de τ para el reporte.
- Multiplicidad: ~10 fuentes posteriores a la geometría ⇒ Bonferroni,
  z = Φ⁻¹(1 − 0,05/(2·10)) ≈ 2,807.
- Tamaño: n_min_f = (z·ŝ_f/δ_f)² con δ_f = fricción/SL (≈ 60 por fuente según el
  censo); antes de n_min el veredicto es `SinEvidencia` (nunca por defecto).

**Algoritmo / código (archivos y anclas, re-grep antes de editar).**
1. `crates/risk-engine/src/lib.rs`: thread-local `ULTIMA_GEOMETRIA:
   Cell<Option<GeometriaIntencion { tp_pct, sl_pct, friccion_pct }>>` junto a
   `ULTIMO_RECHAZO`; se fija tras `tpsl_gate` (`compute_tp_sl_with_target_rr`,
   ~:1025) con la MISMA fricción que usan las puertas de EV/fee; se borra en
   `set_reject_direction`. `pub fn ultima_geometria()`. Cero efecto en la
   decisión (sólo se escribe una celda del hilo).
2. `crates/god-engine-core/src/lib.rs`: campo `libro_sombra: LibroSombra` en
   `GodEngineCore` (memoria fija, sin asignar tras construir). Tras cada
   `evaluate_quantum_order` rechazada: si hay `ultimo_rechazo()` posterior a la
   geometría (suelo_tp_sl, confianza, EV, viabilidad, min_notional, fee,
   orquestador; consejo y ML-gate como fuentes 19 y 20) y `ultima_geometria()`,
   `registrar(IntencionVetada { fuente, moneda, es_largo, entrada: mid, tp_pct,
   sl_pct, friccion_pct, tau_ms: expected_duration_ms, t0_ms })`. Por tick y
   moneda: `observar(moneda, mid, ts)` — O(512) por llamada; optimización
   prevista: contador de abiertas y salida temprana con 0 (sin asignar).
3. Publicación sólo al resolver (raro): `sombra_veto_{n,media_r,se_r,veredicto}`
   por fuente con nombres de tabla estática (sin `format!` en la ruta caliente).
   Ninguna decisión lee claves `sombra_*` (contrato por grep).
4. Contrato `libro_sombra_e2e_contract.rs`: (a) pureza bit a bit — el mismo
   replay con y sin el libro produce trades/PnL idénticos; (b) solución conocida
   — una puerta que veta sobre una cinta sintética con deriva positiva acaba en
   `BloqueaGanadoras` y una que veta sobre deriva negativa en `AhorraDinero`;
   (c) fuentes indexadas por `REJECT_NAMES` sin desplazar los índices 7, 9 y 14
   (F251).

**Uso posterior (no en este átomo).** Un veto con `BloqueaGanadoras`
sostenido en las dos mitades de la muestra pasa al consejo con su evidencia
para relajarlo o retirarlo (con T-1); uno con `AhorraDinero` queda certificado.
## 9. RB-A1 — meta-analítica del impacto histórico de cada cambio (diseño)

**Meta → concepto.** Una regla del operador es medir, no suponer. Hoy el impacto
de un cambio se afirma en el mensaje del commit y no hay una serie que permita
ver tendencias ni atribuir regresiones. Ejemplo real: LG midió el camino
caliente en 134 → 175 µs/evento (+31 %) entre 7ee9c025a y main, «la puerta no
pasa», y no se sabe qué ola lo causó. El sistema convierte la historia de
`main` en un **libro de impacto por cambio** y permite **bisecar** regresiones.

**Fuentes (todas reales, ya en el repo o en `logs/`).**
- Git, en una sola pasada `--first-parent`: sha, fecha, autor, ids de átomo
  (RB-, RET-, Ω, C-F, CL-, LG-, F-A, F1-, QS-, GLM, Sol-R…) y agente inferido;
  líneas ± por crate y por clase (producción, tests, docs, scripts) con `-U0`;
  delta de funciones de test (`#[test]`, `#[tokio::test]`) por commit.
- Cómputo: clausura de dependencias de los paquetes del núcleo en cada versión
  de `Cargo.lock` (caché por blob), coste de build por agente y día desde
  `logs/semaforo.log` (y, desde RB-A1b, por commit: el semáforo anotará
  `head=<sha>` y la rama), mediana µs/evento del medidor CL-60 cuando se mida.
- Salud del código: sitios de pánico de producción por paquete desde el TSV de
  RB-P1 (serie desde hoy), tests por crate.
- Financiero: cobertura del oráculo T-1 (k/144) declarada en los commits y en
  los registros archivados (`docs/audit/**/*t1*.txt`); desde RB-A1c, una línea
  `[METRICAS]` estable en los replays (ROI neto, operaciones, PF, DD, Sharpe)
  atribuida al sha compilado.

**Salida.** `docs/audit/meta_analitica/libro_impacto.tsv` (una fila por commit
de primer padre) y `resumen.md` (deltas por átomo y agente, peores regresiones y
mejores mejoras, tendencias). Regenerable de forma determinista; caché fuera
del control de versiones.

**Atomización.** RB-A1a: el libro desde git + `Cargo.lock` + T-1 de los
mensajes + semáforo por día, con contratos sobre un repositorio git temporal
real (sin mocks). RB-A1b: el semáforo registra `head` y rama ⇒ coste de build
por átomo. RB-A1c: métricas financieras estables de los replays y un bisector
automático (`git bisect run` con el medidor o el replay como juez) para
atribuir regresiones como la del +31 %.

## 10. RB-N1 — el reloj del host vuelve a vivir (entregado el 2026-11-10)

**Síntoma (corrida demo a3d, desde 2026-11-09 01:03, ~32 h).** Una sola
apertura. 97 812 eventos `LATENCY PANIC` con entradas bloqueadas, 85 190
`LATENCY_PANIC` (scalp omitido) y rachas de `KILL_SWITCH`. La componente
«red+reloj» de la edad del dato rondaba 7 s y la deriva D (pared − proyección
monótona) fue de −1,4 s a −4,9 s. El log mostraba «Sincronizador de generación
1 superseded» al arrancar, lo mismo con la generación 2, y nunca una muestra.

**Cuantificación (añadida tras medir; corrige la lectura causal).** Los
97 812 pánicos son el 0,87 % de los ~11,3 M eventos de la corrida, con un pico
del 4 % en el decil 6 del log. La única apertura llegó en el decil 1, y los
deciles 2 y 8, sin ningún pánico, tampoco abrieron nada. El reloj muerto es un
defecto real (puerta de latencia contaminada y offset de firma congelado),
pero **no es la causa** de que la Demo casi no opere. El estrangulador
medido está en la decisión: GLM ya lo tiene en su Fase B (el ML-GATE veta el
ensamble con ml 0,17-0,22, más de 2 800 entradas vetadas en a3).

**Causa raíz.** La guardia HOST-016 (97996eba0, 2026-09-17) guardaba
`my_gen = fetch_add(1)` (valor PREVIO) y salía con `contador > my_gen`,
verdadero desde su primer chequeo. Toda generación, también la única, se daba
por reemplazada, y el `select!` cancelaba el bucle Kalman antes de su primera
muestra. `server_time_offset_ms` (firma de órdenes) y
`server_time_correction_ms` (edad del dato, I-5) quedaron congelados en el
arranque casi dos meses. Los contratos I-5 eran de funciones puras y pasaban:
el bucle nunca se ejecutaba en vivo.

**Física medida el 2026-11-10, independiente del motor.**

| Magnitud | Valor |
|---|---|
| Windows Time | «not synchronized», fuente «Local CMOS Clock», ninguna sincronización registrada |
| Edad real de eventos del testnet (offset de servidor medido al momento) | aggTrade p50 162 / p90 186 / p99 236 ms; depth5@100ms p50 157 / p99 193 ms |
| Deriva del monótono (QPC) frente a Binance, 30 min, 136 muestras | −19,0 ppm (IC95 ±5,2) ⇒ −1,6 s/día, −4,9 s en 72 h |
| Deriva del reloj de pared frente a Binance | −51,7 ppm (IC95 ±5,9) ⇒ −4,5 s/día; offset servidor−pared de ~290 a ~172 ms en 30 min |

De los ~7 s de «red+reloj» de a3d, la red explica ≈0,2 s; el resto es reloj
sin corregir. Riesgo derivado: con el offset de firma congelado y la pared a
52 ppm, el sello de las órdenes cruza +1 000 ms frente al servidor (rechazo
−1021) en ~5,4 h. Los logs de a3 no lo muestran todavía, pero es estructural.

**Arreglo (`execution_engine::ntp`, host sólo como cliente).**
1. `SincronizadorUnico`: `reclamar()` devuelve la generación DESPUÉS del
   incremento; `correr_mientras_vigente()` sólo detiene un relevo real.
2. `EstadoReloj` común a todas las generaciones y anclado por el HOST
   (`fijar_ancla(epoch_baseline_ms, instant_baseline)`): la corrección se mide
   contra la proyección que consume la edad del dato. Absorbe así el error del
   offset de arranque, y el relevo demo→mainnet no la reinicia. Con un ancla
   propia por generación, la deriva acumulada volvería de golpe a la edad del
   dato justo al pasar a Producción: defecto latente del diseño anterior, que
   el bucle muerto ocultaba.
3. La proyección se evalúa en el punto medio MONÓTONO del viaje (sin sesgo
   −RTT/2), y el offset de arranque en el punto medio de pared.
4. Un observador fuera de la ruta caliente deja en el log «🕐 [NTP-SYNC]
   Generación N viva — muestra #k» (las 2 primeras y una por hora), más un
   aviso si la puerta de RTT descarta todo durante un cuarto de hora.

**Certificación.** `crates/execution-engine/tests/rb_n1_reloj_vivo_contract.rs`:
6/6 en tres repeticiones, más la sonda viva contra el testnet. La sonda corre
el bucle de producción con la guardia de producción: 4 muestras en 50 s, y
partiendo de un ancla con el error de un arranque caído a la hora local, la
corrección convergió a 185 ms frente a 176 ms de un offset medido aparte.
Mutaciones: A (guardia con valor previo) la detectan 2 contratos, B
(proyección en la llegada) 2, y C (re-anclar sin reiniciar el filtro) 1. Una
corrida de B destapó que el test de conducta dependía de la granularidad del
temporizador de Windows (~15,6 ms); quedó reescrito sin temporizador. I-5 5/5
y Kalman 3/3. rustfmt y clippy limpios en las líneas cambiadas de
`execution-engine` y del binario `god_engine`.

**10.5 RB-N2 (propuesto, con Línea I por I-2b/I-5).**
- El umbral de pánico es un GEN (`latency_ms_panic_threshold`, semilla
  `1000·π`, hoy ≈6 945 ms). En backtest no hay latencia de red, así que la
  evolución no lo puede identificar: deriva sin señal. Debe salir del genoma
  y derivarse de la distribución viva de «red+reloj» (cuantil alto robusto o
  ajuste de colas EVT), separando el exceso de reloj (que el sincronizador ya
  corrige) del de red.
- El Kalman usa R = (RTT/2)² ≈ 14 400 ms², pero la dispersión real del offset
  ronda 16 ms (R ≈ 256). Resultado: ganancia ≈0,006, constante de ~42 min y
  ~60 ms de retraso estable ante 19 ppm. La evolución es un modelo de 2
  estados (offset + deriva, sin retraso ante una rampa) con R adaptativa por
  innovaciones (Sage–Husa) y filtro de mínimo retardo como el de NTP.
- La puerta fija `RTT > 350 ms` es un valor duro. Debe ser relativa al RTT
  mínimo observado del endpoint: `demo-fapi` mide ~790 ms, y un cliente que
  sincronizara ahí descartaría todas las muestras.
- Al operador (requiere administrador): activar la sincronización de w32time.

**10.6 RB-G1 — censo de alcanzabilidad desde el host.** En
`docs/audit/rb_g1_censo_alcance_host_2026-11-10.txt` hay un censo léxico
(salvedades en su cabecera). `evaluate_omniscient_living_graph`,
`evaluate_quantum_multi_slot` y `synthesize_quantum_multi_slot_emissions`
tienen 0 llamadas en código de producción. 62 módulos (casi todos motores de
las olas Ω220) sólo los usa `omniscient_living_graph.rs`, y 116 no tienen
usuario de producción. Los 44 vetos del grafo vivo, y sus rojos de CI, viven
en código que la Demo no ejecuta. Conectarlo tal cual sería dañino: 44 vetos
encadenados con FPR de 1 % cada uno dejan P(sin veto) ≈ 0,99^44 ≈ 64 %, y
con 5 % ≈ 10 %. La vía correcta es la puerta anti-cosmética: cada motor
demuestra valor fuera de muestra neto de fees y entra como evidencia
ponderada (RB-V1, LibroSombra), no como eslabón de una cadena de precedencia.
Siguiente paso de RB-G1: pasar del censo léxico a un contrato sobre el grafo
de llamadas del host.

## 11. RB-V1a — los vetos posteriores al riesgo, juzgados en sombra (2026-11-10)

**Pregunta.** La Demo casi no opera y el estrangulador medido es el ML-GATE
(más de 2 800 vetos en a3; ml 0,17-0,22 en BTC). ¿Bloquea operaciones que
pagan o que pierden? Un veto es una predicción («esta intención pierde») que
hasta hoy no recibía nunca su prueba contrafactual.

**Qué se construyó** (ningún camino de decisión lo lee):
- `risk_engine::medida_sombra` (nuevo). `MedidaOrden::geometria_sombra()`
  deriva TP, SL y fricción de ida y vuelta de la medida que el riesgo deja
  pendiente al validar (D-744, CL-106):
  - `sl = (sl+f) / ((sl+f)/sl)`;
  - `f = (sl+f) − sl`, con una tolerancia de redondeo de 4ε;
  - `tp = w·sl + f`.

  `RiskEngine::medida_validada()` la lee sin consumirla.
- `audit_engine::shadow_ledger`:
  - veredicto de Hoeffding con el ancho conocido de cada resultado,
    `(tp+sl)/sl`, válido para todo n;
  - prefijo denso con borrado por intercambio: `registrar` y `observar`
    recorren sólo las abiertas (antes, 512 casillas ≈ 29 KB por llamada) y
    `abiertas()` es O(1).
- `god_engine_core::sombra_vetos` (nuevo):
  - fuentes consejo (19), ML-GATE sin modelo (20) y ML-GATE sin ventaja (21),
    por encima de las `REJ_*` (0..19, reservadas para RB-V1b); el veto se
    atribuye al primero de la cadena;
  - α de familia 0,05 por Bonferroni ⇒ δ = 0,05/6 ≈ 0,0083 por cola;
  - `observar` es O(1) si la moneda no tiene abiertas y no asigna;
  - avisos `🕵️ [SOMBRA]` en escala log₂ de las resoluciones y al cambiar el
    veredicto, por la cola sin bloqueo de la telemetría (búfer fijo de 512 B).
- Núcleo, 5 puntos: `pub mod`, el campo junto a `diag_ml_vetoes`, su init, la
  observación al inicio de `process_tick_dual` con el mid del evento, y el
  registro tras la decisión (`if !(aprobado_por_consejo && ml_gate_ok)`).

**Matemática.** El resultado de cada intención es
`r = (bruto − f)/sl ∈ [−(sl+f)/sl, (tp−f)/sl]`. Para variables acotadas no
idénticas, Hoeffding da
`P(|r̄ − E r| ≥ √(ln(1/δ)·Σaᵢ²/2)/n) ≤ δ` por cola. Un veto que sólo bloquea
perdedoras (todo SL, R = −1,2, ancho 3) queda certificado «ahorra dinero» en
n = ⌈4,5·ln(120)/1,44⌉ = 15. Se descarta z·SE porque es asintótico y obliga a
fijar un n mínimo arbitrario.

Límites declarados:
- supone independencia, y las monedas correlacionadas no lo son;
- resuelve con el mid, no con el último precio ni con el mark;
- no modela el impacto propio ni el rechazo de la orden.

**Certificación.**
- **Contratos:**
  - audit: QS-R4 9/9 (el prefijo denso no cambia la conducta) y Hoeffding
    4/4. Cubren la forma cerrada, poblaciones conocidas, una cobertura Monte
    Carlo de 2 000 réplicas de un juego justo con veredictos falsos ≤ 2δ, y la
    certificación en el n exacto;
  - riesgo: 3/3, más una orden validada REAL 1/1 (TP 0,020, SL 0,010,
    fricción 0,00082 = 2·0,0004 + 2·0,00001, idéntica a lo que mide la cuenta
    al abrirla), más C-F5c 2/2;
  - núcleo: 4/4 unitarias y contrato 4/4 (cableado, pureza por fuente y
    conducta del `process_tick_dual` real: TP sólo con el mid de su moneda;
    vencimiento a τ con +0,2 R).
- **Mutaciones:** M1 (la sombra no avanza) la detectan 2 pruebas; M2 (TP sin
  fricción) 1; M3 (radio 10× menor) 4; M4 (sin control de solape) 1.
- **Estilo:** clippy y rustfmt limpios en mis archivos y regiones. En el
  núcleo, mi bloque tiene la forma de rustfmt con la sangría de sus vecinos:
  la región de `profile_node!` arrastra una sangría previa.
- **Replays de cinta real** (`#[ignore]`, `crates/backtest-engine/tests/rb_v1a_*`):
  8 ventanas independientes de 100 000 ticks reales:
  - BTC, 31-08-2026, 6 ventanas contiguas, 13:39→23:59 UTC (10,35 h);
  - ETH, 31-08-2026, 2 ventanas, 17:42→23:59 UTC (6,3 h).

  Corre el núcleo de producción con el genoma de referencia, omni neutro y
  sin modelos del roster. Detalle en
  `docs/audit/rb_v1a_certificacion_sombra_2026-11-10.txt`.
  - **Embudo:** 2 508 vetos del ML-GATE y 2 042 del consejo; el riesgo
    rechaza por `sonda_ocupada` y `confianza`.
  - **Operaciones:** 8 en total, todas la sonda de arranque en frío, y las 8
    pierden. PnL bruto de −0,06 a −0,22 USDT y neto de −0,08 a −0,30 USDT
    por operación; comisión del 0,1 % de ida y vuelta.
  - **Sombra:**
    - ML-GATE sin modelo: 10 resueltas (9 por τ, 1 SL), R neto medio ≈ −0,71;
    - consejo: 2 resueltas (τ), −0,31 R;
    - veredicto «sin evidencia»: con n = 10 el radio ronda 1,5 R;
    - miles de solapes, porque se admite una intención abierta por fuente,
      moneda y lado.

  Lectura: sin modelos, el ML-GATE sólo deja pasar la sonda (B3.25) y lo que
  bloquea tampoco paga, de momento. El estrangulador de la Demo, el ML-GATE
  «sin ventaja» con modelo, se mide en la Demo.
- **Pureza A/B:** huella del replay idéntica bit a bit entre `main` sin el libro y la rama con
  él, en la ventana BTC v6 (100 000 ticks, donde la rama registró y resolvió
  5 vetos):
  - las dos dan 1 operación, `net_pnl` 0xbfd35817b2277730 y `final_capital`
    0x408f3d94fd09bb11, y también coinciden el PnL bruto y las comisiones;
  - tardan 394 s frente a 397 s: el coste del libro no se distingue del ruido.

**Hallazgos de la certificación** (para sus dueños):
1. **Un replay sin símbolo registrado no produce ni una intención.**
   `sync_reality_fees_for_coin` devuelve falso, `allow_entries` queda en falso
   y todo se bloquea antes del riesgo. Mis primeras corridas (31 días de ZIL,
   17,5 h de BTC y 20,7 h de ETH) dieron 0 operaciones y 0 llamadas al riesgo
   por eso; el certificador ya registra el símbolo de la cinta. Cualquier
   herramienta que use `run_booktick_replay` debe hacer lo mismo, o evaluará
   genomas que nunca operan: fitness sin señal.
2. **Specs de replay con valores fijos.**
   - `backtest_windows` registra tick 0,01 y paso 0,001 para CUALQUIER símbolo.
   - `get_official_binance_spec` es una tabla fija con un fallback (0,01/0,001)
     absurdo para monedas de céntimos: ZIL cotiza a 0,0029.

   Propuesta: una instantánea de `exchangeInfo` como fuente única de specs
   para replays.
3. **Coste por tick del núcleo en debug:** 0,65 ms con las entradas bloqueadas
   frente a 7,9 ms con entradas (12×). El presupuesto de la ruta caliente se
   mide en release; la proporción indica dónde mirar.
4. **CI con `--no-fail-fast` (RB-C1): 7 rojos en el shard núcleo.**
   - AGY: `f69_002`, `f72_002` y tres de `f99_002`. Son precedencias de veto:
     el veto Fracton de W29 y el Starobinsky ganan la carrera.
   - GLM-223: dos de `ricci_q_publicacion_contract` (publica NaN y R = −66,1).

**Siguiente.**
- RB-V1b: llevar a la sombra los rechazos del riesgo posteriores a la
  geometría (`suelo_tp_sl`, EV, fee, viabilidad). Requiere la geometría de la
  intención rechazada (thread-local `ULTIMA_GEOMETRIA`, §8.1).
- Leer el veredicto vivo en la Demo (GLM): `grep "\[SOMBRA\]"` en el log de la
  corrida.

## 12. RB-V1b — los rechazos del riesgo en sombra y un veredicto siempre válido (2026-11-10)

**Qué cambia** (nada lo lee ninguna decisión):
- `risk_engine::medida_sombra`:
  - `RiskEngine::geometria_evaluada(coin)` publica TP, SL, fricción de ida y
    vuelta y τ de la evaluación en curso, etiquetados con su moneda. Se borra
    al empezar cada `evaluate_quantum_order` y se fija dos veces dentro de
    `evaluate_single_intent`:
    - con la geometría del horizonte, que es la que juzga el suelo TP/SL;
    - con la geometría final de la orden, tras la validación de objetivos.
  - En producción las intenciones no llevan objetivos explícitos
    (`tp_price_target`/`sl_price_target` sólo aparecen declarados), así que las
    dos geometrías coinciden.
  - Sustituye la derivación de RB-V1a desde `MedidaOrden`: es exacta en origen
    y sirve también a los rechazos, que no dejan medida.
- `god_engine_core::sombra_vetos`:
  - 8 fuentes nuevas con su índice `REJ_*`: EV, fee_impact, min_notional,
    orchestrator, suelo_tp_sl, confianza, viabilidad y sin_evidencia.
  - `fuente_juzgable(rechazo)` decide cuáles se juzgan. Quedan fuera los
    rechazos anteriores a la geometría (spec, exposición, correlación, entradas
    inválidas, sonda ocupada, estrategia falsada), porque no traen operación
    que juzgar, y el riesgo duro (drawdown).
  - El núcleo registra justo tras la única `evaluate_quantum_order` del tick y
    antes de deliberar; el registro de RB-V1a usa la misma geometría.
- `audit_engine::shadow_ledger`: el veredicto de Ville sustituye al de Hoeffding.

**Defecto corregido en RB-V1a: mirar en cada resolución.**
- Hoeffding da una cota válida para un n FIJADO de antemano. El núcleo vuelve
  a evaluar el veredicto en cada resolución y lo anuncia en cuanto cruza: es
  parada opcional. Por la ley del logaritmo iterado, un radio √(c/n) acaba
  cruzándose con probabilidad 1.
- Medido con un juego justo (E[r] = 0 en cada intención, geometrías distintas),
  δ = 0,05/6 por cola y mirando en cada n hasta 4 000: Hoeffding se equivoca en
  el **3,10 % ± 0,27 %**, frente a un nominal del 1,67 % (5,3σ).
- El veredicto nuevo, mirando en cada resolución con el libro real (1 500
  réplicas × 1 000 miradas): **0,33 %** (cota 1,67 %). En la configuración de
  producción (α = 0,05/22 por cola): **0,07 %** (cota 0,45 %).

**Matemática del veredicto.**
- `x = r/ancho` vive en un intervalo de ancho 1, con `ancho = (tp+sl)/sl`
  fijado al registrar. Por el lema de Hoeffding, si `E[x | pasado] ≤ 0`,
  entonces `E[exp(u·x − u²/8) | pasado] ≤ 1` para toda u ≥ 0.
- El e-proceso es `M⁺_n = (1/J)·Σ_j Π_i exp(u_j·x_i − u_j²/8)`, con
  `u_j = 4·2^(−j)`, j = 0..8. `M⁻` es lo mismo con `−x`.
- Bajo cada hipótesis nula («lo vetado no paga» o «no pierde»), M es una
  supermartingala no negativa con M₀ = 1. La desigualdad maximal de Ville da
  `P(∃n: M_n ≥ 1/α) ≤ α`.
- La rejilla cubre todas las medias normalizadas posibles (u* = 4·x̄ y |x̄| < 1)
  hasta x̄ ≈ 1/128. Su mejor apuesta conserva al menos el 83 % del crecimiento
  óptimo `2·x̄²`, a cambio de ln 8 = 2,08 nats de mezcla.
- Ejemplo: un veto que sólo bloquea perdedoras se certifica en **n = 26** con
  α = 0,05/22, el valor exacto de la forma cerrada. Con el radio inválido de
  Hoeffding, y el mismo α, habría sido n = 20.
- El log muestra la evidencia en logaritmos: `paga e^…, pierde e^… (umbral
  e^6,1)`.

**Certificación.**
- **Contratos:**
  - audit: QS-R4 9/9 y Ville 5/5 (forma cerrada, poblaciones conocidas,
    potencia en el n exacto, validez mirando en cada resolución e inflación
    medida de Hoeffding);
  - riesgo: 2/2 unitarias y 3/3 de integración. Cubren tres casos: la orden
    validada publica su geometría exacta (τ y fricción iguales a los que mide
    la cuenta); un rechazo por confianza la conserva; un rechazo anterior a la
    geometría no publica ninguna y borra la anterior. C-F5c 2/2.
  - núcleo: 5/5 unitarias y 5/5 de contrato (cableado de los dos registros,
    pureza, conducta del `process_tick_dual` real).
- **Mutaciones:** 5/5 detectadas:
  - M1 (castigo con el signo cambiado en «pierde»): 4 pruebas;
  - M2 (castigo u²/16: deja de ser supermartingala): 3;
  - M3 (no se borra la geometría al empezar): 1;
  - M4 (`fuente_juzgable` acepta fuentes del núcleo): 1;
  - M5 (la geometría final es la del horizonte): 2.
- **Estilo:** clippy y rustfmt limpios en mis archivos y en las regiones tocadas del núcleo y del riesgo.
- **Replays de cinta real:** las mismas 8 ventanas de 100 000 ticks que RB-V1a (BTC 31-08-2026
  13:39→23:59 UTC; ETH 17:42→23:59 UTC), con el núcleo de producción. Detalle
  en `docs/audit/rb_v1b_certificacion_sombra_2026-11-10.txt`.
  - **Embudo:**
    - vetos: 2 508 del ML-GATE y 2 042 del consejo;
    - rechazos del riesgo: 689 por confianza (posteriores a la geometría, se
      juzgan) y 1 300 por sonda ocupada (anteriores, no se juzgan).
  - **Sombra, fuente nueva `riesgo:confianza`:** 11 resueltas, todas por τ,
    con R neto medio ≈ −0,60.
  - **Sombra, fuentes de RB-V1a:** ML-GATE sin modelo 10 resueltas (≈ −0,71) y
    consejo 2 (−0,31).
  - **Evidencia:** ninguna fuente pasa de e^0,1 frente al umbral e^6,1, así
    que no hay veredicto todavía.
  - **Operaciones:** 8, todas la sonda de arranque en frío, y las 8 pierden
    (neto de −0,08 a −0,30 USDT por operación).
  - **Lectura:** lo que rechaza la confianza, lo que vetan el consejo y el
    ML-GATE, y lo que se opera sale por tiempo y no cubre la fricción del
    0,1 %. La pregunta abierta no es qué puerta relajar, sino qué horizonte y
    qué geometría pagan la fricción.
- **Pureza A/B:** huella idéntica bit a bit entre `main` 206256e2b (con RB-V1a y sin RB-V1b) y
  la rama, en BTC v6, donde la rama registró y resolvió 8 intenciones de tres
  fuentes:
  - las dos dan 1 operación, `net_pnl` 0xbfd35817b2277730 y `final_capital`
    0x408f3d94fd09bb11;
  - tardan 412,8 s frente a 412,6 s.

**Siguiente.**
- Leer los veredictos vivos en la Demo (GLM: `grep "\[SOMBRA\]"`).
- El SL se cobra en la barrera aunque el mid salte más allá: ligeramente
  optimista para «bloquea ganadoras». Es despreciable en monedas líquidas, pero
  queda declarado.
- RB-V1c: una ablación contrafactual de verdad (AIPW), para cuando haya
  operaciones reales con las que compararla.

## 13. RB-V1c — promesa frente a realidad: la puerta de EV del riesgo (2026-11-10)

**Diagnóstico.** Sale de la lectura del código; las cintas lo confirman.
1. **Suelo del stop.** `MAX_FRICTION_SHARE_OF_RISK = 0.65` (`genome.rs`) fija
   `sl ≥ f/0,65`, unos 15,4 pb con la fricción de 0,1 %. Su propio comentario
   dice que «es una política elegida, no una constante física». La fricción
   puede ser el 65 % del stop, y una operación que vence sin moverse cobra
   R ≈ −f/sl ≈ −0,65.
2. **Geometría.** `compute_tp_sl` fija `TP = SL·min_rr_for(0,40, f, sl)`. Con
   f/sl = 0,65 sale RR = 1,5 + 1,625 = 3,1. El stop es `k·σ(τ)`, con k ≈ 1.
   Para una difusión sin deriva en el horizonte τ:
   - P(tocar el SL a 1σ antes de τ) ≈ 2·(1 − Φ(1)) ≈ 32 %;
   - P(tocar el TP a 3,1σ) ≈ 0,2 %;
   - el ~68 % restante sale por τ.
3. **Puerta de EV.** `expected_value_pct = p·TP − (1−p)·SL`, con p la
   confianza calibrada de DIRECCIÓN (0,5 en la sonda). Tiene dos fallos:
   - es un modelo binario que ignora las salidas por τ;
   - confunde «acertar la dirección» con «tocar el TP antes que el SL».

   La probabilidad de primer toque (`probabilidad_tocar_sl_antes_de_tp`) se
   calcula, pero desde CL-100 sólo es telemetría.

   Ejemplo, la sonda (p = 0,5 y TP = 3,1·SL): promete +0,45 R netos y las
   cintas dan ≈ −0,6 R.

**Qué cambia** (sólo observa):
- `RiskEngine::promesa_evaluada(coin)` publica el EV bruto de la puerta.
  Se borra al empezar cada evaluación y se fija tras calcularlo.
- `SombraVetos` registra la promesa con cada intención y la convierte a R
  neto con la misma fricción y el mismo stop con que el libro mide lo
  realizado.
- `LibroSombra` acumula por fuente la brecha realizado − prometido y un
  e-proceso de Ville sobre `x = (r − promesa)/ancho`. La promesa es fija al
  registrar, así que la brecha conserva el ancho de r.
  - `veredicto_promesa` dice «PROMETE DE MÁS» o «promete de menos».
  - Usa el mismo α por cola que los veredictos de lo vetado, como familia
    aparte: el error total de las dos familias es ≤ 10 %.
- `EProcesoVille` se extrae como estructura reutilizable. También servirá
  para la certificación viva Demo → Producción del mandato §31.4.

**Certificación.**
- **Contratos:**
  - audit: QS-R4 9/9, Ville 5/5 (mismos valores bit a bit tras el refactor) y
    promesa 4/4. Cubren la forma cerrada, la promesa optimista certificada en
    n = 13 exacto, cero falsos con una promesa calibrada mirando en cada
    resolución, y la validación de entradas;
  - riesgo: la orden validada promete exactamente 0,9·0,02 − 0,1·0,01 = 0,017;
    un rechazo por confianza, anterior al EV, no promete nada;
  - núcleo: conversión a R y aviso al cambiar el veredicto de la promesa.
- **Mutaciones:** 5/5 detectadas:
  - M1 (brecha con el signo cambiado): 2 pruebas;
  - M2 (una intención sin promesa cuenta como promesa 0): 2;
  - M3 (promesa a R sin descontar la fricción): 1;
  - M4 (la promesa no se borra al empezar): 1;
  - M5 (promesa del doble): 1.
- **Estilo:** clippy y rustfmt limpios en mis archivos y en las regiones tocadas del núcleo y del riesgo.
- **Replays de cinta real:** las mismas 8 ventanas (BTC 31-08-2026 13:39→23:59 UTC; ETH
  17:42→23:59 UTC). Detalle en
  `docs/audit/rb_v1c_certificacion_promesa_2026-11-10.txt`.
  - **ML-GATE sin modelo** (10 intenciones con promesa): el riesgo prometió de
    media **+0,89 R** y se realizaron **−0,71 R**, una brecha de ≈ −1,6 R.
  - **Consejo** (2): promesa +0,37 R frente a −0,31 R realizados.
  - **Rango:** en las 12 intenciones con promesa, prometió entre +0,37 y +1,05 R
    y se realizaron entre −0,31 y −1,30 R.
  - **`riesgo:confianza`:** no lleva promesa, porque se juzga antes del EV.
  - **Evidencia:** la mayor es e^0,64, lejos del umbral e^6,1. Con una brecha de
    este tamaño (x ≈ −0,4), el veredicto «PROMETE DE MÁS» llegaría hacia
    n ≈ 25-30 resoluciones: en la Demo, cuestión de horas.
- **Pureza A/B:** huella idéntica bit a bit entre `main` 0a365893e (con RB-V1b, sin RB-V1c) y la
  rama, en BTC v6, donde la rama registró 8 intenciones de tres fuentes (5 con
  promesa): 1 operación, `net_pnl` 0xbfd35817b2277730 y `final_capital`
  0x408f3d94fd09bb11 en las dos.

**Propuesta RB-E1** (cambia decisiones: requiere el consejo y la evidencia de
la Demo).
- Sustituir la puerta binaria por el EV de triple barrera bajo el modelo:
  - deriva de la señal (`μ√τ/σ = Φ⁻¹(q)` con q la probabilidad de dirección);
  - σ de la ley de difusión;
  - barreras TP y SL, y horizonte τ;
  - probabilidades de primer toque y salida por τ de la serie de doble
    barrera.
- Derivar el suelo de fricción de «EV ≥ umbral», no de una cuota fija de 0,65.
- Antes de proponerlo: el veredicto «PROMETE DE MÁS» en la Demo y una A/B en
  las cintas.