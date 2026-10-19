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
| RB-V1 | **LibroSombra** (censo §3 prioridad 2: F250/F272/F400, sin dueño): qué vetos destruyen valor | 2 | Después de RB-A1; diseño en §8 (región del núcleo que AGY edita en sus olas: se coordina antes) |
| RB-S1 | Antifragilidad: cuarentena en caliente por motor votante | 4 | Pendiente (coordinar con dueños del orquestador) |

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