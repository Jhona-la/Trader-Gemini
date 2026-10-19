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
| RB-H1 | Higiene de `target/` por **presupuesto** + lanzadores con un solo target | 5 | **Entregado** (§3) |
| RB-M1 | Fugas de memoria en runtime del motor vivo | 5 | Siguiente |
| RB-V1 | **LibroSombra** (censo §3 prioridad 2: F250/F272/F400, sin dueño): qué vetos destruyen valor | 2 | Pendiente |
| RB-S1 | Antifragilidad: cuarentena en caliente por motor votante | 4 | Pendiente (coordinar con dueños del orquestador) |
| RB-P1 | Trinquete de pánicos (`unwrap`/`expect`/`panic!` en producción) | 5 | Pendiente |

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

**Para medir fugas en vivo hay que mirar Private Bytes, no Working Set**: el
host compacta el working set bajo presión (`god_engine.rs:2311-2323`). Los
núcleos de examen del demonio (≈25 por ronda cada 3 min, ~39 MB cada uno) son
churn de asignación, no fuga.
