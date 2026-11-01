# Plan de línea Claude QSR — revisión desde la base (2026-11-01)

**Autor:** Claude QSR (Opus 5.5), sesión del 2026-11-01.
**Base revisada:** `origin/main = 4eae74027` al empezar; `33dfe1064` tras el heartbeat.
**Relación con el plan canónico:** el plan de coordinación sigue siendo
[`docs/PLAN_MAESTRO_FASES_2026-10-22.md`](PLAN_MAESTRO_FASES_2026-10-22.md) (V33).
Este documento es un plan de línea: aporta lo medido hoy, los hallazgos de esta
auditoría y a quién le toca cada uno. No es una fuente paralela de verdad.
Las reservas vivas están en `AGENT_SYNC.md`.

## 1. Estado verificado al empezar (2026-11-01 ~14:10–14:30 Bogotá)

| Área | Medido | Consecuencia |
|---|---|---|
| Ramas remotas | `main`, `claude/project-thread-apa725` (PR #151 MERGED, tip ancestro de main), `claude/project-thread-lzkhpo` (PR #154 draft; Línea C la fusionó a las ~14:20) | Borradas las dos tras comprobar PR MERGED, tip ancestro de `origin/main` y cero commits exclusivos. GitHub queda sólo con `main`. |
| CI de `main` | Último código integrado `84e95ebaf`: Unit y Replay en verde. `1710deaba` falló en Unit (pánico en el constructor de `ventana64()`, cazado por el trinquete RB-P1) y `84e95ebaf` lo corrigió. El merge `051d0269d` (PR #154) estaba en curso. | No es cierto que «todos los commits fallen»: el rojo de 1710deaba ya se arregló. Cada push se valida por su SHA. |
| Checkout compartido | `main` local 4 commits adelante y 43 detrás de `origin/main`; 647 archivos con cambios sin commit de otros agentes | No se toca. Esta línea trabaja en `../tg-claude-qsr` (rama `claude/qsr-2026-11-01`). |
| Worktrees | 10 worktrees en `detached HEAD` bajo `logs/agent-fabric/` y `../tg-agent-fabric` | Son de agent-fabric; no se retiran sin su dueño (AGENTS.md, regla de worktrees). |
| RAM | 23,38 GiB visibles, **0,79 GiB libres** sin compilación propia. Mayor consumo: hosts de agentes (Kimi Code 0,74+0,37, ChatGPT 0,63, Hermes 0,58, Edge, Claude ×3, Codex, ZCode ×2, WorkBuddy) | La RAM la ocupan las apps de agentes, no el bot. Ver §6. |
| Disco C | 255,9 GB libres | Sin presión de disco; ver §6 para `target/`. |
| Leases vivos | Kimi AGENT-01 (A4-T2): `entry_dispatch.rs`, `execution_evidence.rs`, `executor.rs`, `symbol_registry.rs`, `symbols.rs`, `god_engine.rs`, `host_oms.rs` y sus contratos. Codex QRA: `god-engine-core/src/lib.rs` | Esta línea no edita esos archivos; sus hallazgos se entregan a los dueños. |

## 2. La meta en números

La meta del dueño es duplicar el capital cada 72 h. Es una hipótesis a
demostrar fuera de muestra y después de fees, funding y slippage; no un
resultado.

- `ln 2 = 0,6931` nats cada 72 h = 0,2310 nats/día = **+25,99 % simple diario**,
  **+0,967 % por hora** o **1,60 pb por minuto**, todo sobre el capital.
- El sistema envía hoy todas las entradas como IOC taker (`force_maker = false`,
  B3.29/CL-14) y el gate modela ida y vuelta taker: `2·taker + 2·slip`
  (`risk-engine/src/tp_sl.rs::roundtrip_friction`). Con VIP0 USDT-M
  (taker 0,05 %, maker 0,02 % por pierna) la fricción de comisiones es
  10 pb de nocional por operación antes de slippage.

Retorno bruto que cada operación debe capturar sobre su nocional para llegar a
la meta (neto por ciclo sobre capital = 0,967 %/h ÷ ciclos):

| Ruta | nocional/capital | 10 ciclos/h | 30 ciclos/h | 60 ciclos/h |
|---|---|---|---|---|
| taker–taker (10 pb) | 1 | 19,7 pb | 13,2 pb | 11,6 pb |
| taker–taker | 5 | 11,9 pb | 10,6 pb | 10,3 pb |
| maker–taker (7 pb) | 2 | 11,8 pb | 8,6 pb | 7,8 pb |
| maker–maker (4 pb) | 2 | 8,8 pb | 5,6 pb | 4,8 pb |

Lectura: con la ruta actual, por mucho apalancamiento que haya, cada operación
necesita capturar al menos ~10,3 pb brutos sólo para cubrir comisiones y la
meta. Cambiar una pierna a maker baja el listón 3 pb y las dos, 6 pb; por eso
la ejecución pasiva (con su cola FIFO y su selección adversa medidas, como pide
el plan canónico) pesa más que cualquier teoría nueva. Ningún número de esta
tabla es una promesa: las tasas reales por símbolo se leen de
`/fapi/v1/commissionRate` y el slippage y la latencia se suman aparte.

## 3. Fases de esta línea (dentro de las del plan canónico)

El orden es el que pide el dueño: metas → conceptos → matemática → código, de lo
más importante a lo menos, y cada paso con prueba.

1. **F0 — Sincronización y recursos** (hecho hoy): heartbeat, ramas, CI,
   medición de RAM y disco, leases. Salida: §1 y §6.
2. **F1 — Auditoría de solo lectura con verificación adversarial** (hecho hoy):
   cuatro dimensiones —fees de extremo a extremo; config y lib; secuencia de
   arranque y paridad de balance; motores muertos y selección de símbolos—, cada
   una revisada por un escéptico que intenta refutarla. Salida: §4.
3. **F2 — Un átomo por vez** con test RED antes del arreglo y GREEN después,
   sólo en archivos sin lease, `cargo check --workspace --all-targets` y la
   suite del crate tocado. Salida: §5.
4. **F3 — Entrega a dueños**: cada hallazgo en archivo con lease va al buzón con
   su test en rojo propuesto.
5. **F4 — Siguiente ola**: retomar los hallazgos `PLAUSIBLE` con una medición
   (replay o contrato) antes de tocarlos.

## 4. Hallazgos de la auditoría del 2026-11-01

Método: cuatro auditores de solo lectura sobre `origin/main = 4eae74027` (fees de
extremo a extremo; config y lib; arranque y paridad de balance; motores muertos y
selección de símbolos). Cada hallazgo pasó después por un verificador escéptico
que intentó refutarlo leyendo el código. Resultado: 28 hallazgos, de los que 12
ya estaban registrados (no se repiten aquí: FEE-01/02/04/05/07, QSR-CL-04/06,
BOOT-03/05/06, BT-01, SEL-01) y dos auditores encontraron el mismo defecto
(QSR-CL-01 = RSK-01). Quedan 15: 13 confirmados y 2 plausibles. Las líneas son de
`4eae74027`; antes de editar, vuelve a buscar el símbolo.

| ID | Veredicto | Dónde | Qué pasa | Dueño |
|---|---|---|---|---|
| SEL-02 | CONFIRMADO, ALTO | `god-engine-core/src/bootloader.rs:227`; consecuencia en `lib.rs:3993-4099` | El arranque lee `data/dynamic_config.json` (ruta común) sin mirar `is_testnet`; el host vigila `data/{demo,prod}/` y el daemon nunca escribe ahí, así que la recarga no se dispara. Con el archivo real del operador (testnet, 23-09) un arranque de mainnet pone ATOMUSDT en el slot 0, y el núcleo calcula el régimen GLOBAL de toda la cartera desde el slot 0, que supone BTC. | Kimi V34 B1+B3 ya exige venue coincidente en su WIP local (no en main). El anclaje del régimen por identidad de BTC (no por slot) es de Codex QRA (lease de `lib.rs`). |
| BOOT-02 | CONFIRMADO, MEDIO | `bootloader.rs:194`; `src/host_oms.rs:383-397, 497-515` | El universo nunca se valida contra `exchangeInfo`. La autorización OMS exige comisiones y posiciones de TODO el universo: un símbolo deslistado o ausente en testnet deja `ready=false` y las demás monedas sin operar toda la sesión. | Bootloader: Kimi (B1). `host_oms.rs`: Kimi (lease A4-T2). |
| RSK-01 | CONFIRMADO, MEDIO | `risk-engine/src/lib.rs:1376-1384` | La guarda de liquidación F1-088 usa `maintenance_margin_rate`, que nadie escribe: 1 % fijo para todo símbolo. En Binance el primer tramo es ≈ 1/(2·L_max): 2,5 % a 20×. Con stop del 3 % admitía L = 20 y la liquidación quedaba antes del stop. | **Átomo de esta sesión (§5).** El arreglo completo (publicar la tasa desde `leverageBracket`) toca el ejecutor: Kimi. |
| MUE-02 | CONFIRMADO, MEDIO | `signal-engine/src/orchestrator.rs:594-604` | La calibración RMT de Ω209 lee `rmt_effective_bets_ratio`, que nadie escribe; `random_matrix::effective_bets` no tiene llamador. El boost de ensamble cuenta clones colineales como apuestas independientes (hasta ×1,40). | AGY (Ω209) con Qoder (orquestador). |
| MUE-01 | CONFIRMADO, MEDIO | `god-engine-core/src/lib.rs:7104-7145` | La homología topológica y la compuerta RedTeam corren en cada tick y publican 9 claves que ningún código de producción lee: el «veto pre-trade» no veta nada y cuesta CPU en el camino caliente. | Codex QRA (lease); autor AGY. |
| QSR-CL-02 | CONFIRMADO, MEDIO | `lib.rs:8432-8433, 8504, 8531` | La puerta ML de la rama 13 sigue con la escala de semi-intervalo que F1-052 corrigió sólo en el gate final: los cortos exigen la mitad de ventaja que los largos con base 1/3. | Codex QRA. |
| QSR-CL-03 | CONFIRMADO, MEDIO | `lib.rs:8135-8139, 8235, 8689-8716` | Ramas 11 y 14: el corte trata `|net|` como ΔP = P − 0,5 y el mapeo a confianza como 2·ΔP. Intenciones admitidas por el gen salen con confianza menor que el propio gen. | Codex QRA. |
| BOOT-01 | CONFIRMADO, MEDIO | `src/bin/god_engine.rs:1445-1486, 1704-1764` | Todas las peticiones firmadas del arranque (balance, comisiones, posiciones, modo hedge) usan el reloj local crudo: la hora del servidor se mide después. Ninguna envía `recvWindow`. | Kimi (lease). |
| FEE-03 | CONFIRMADO, MEDIO | genes `vec[68]`, `vec[69]`; `evolution-engine/src/random_forest.rs:72-107` | El deslizamiento que cobran la puerta, la física y la tolerancia IOC son genes: cada candidato se juzga con su propio modelo de coste, y la evolución puede «ganar» bajando lo que se cobra a sí misma. | Congelar o medir: Línea G / Fable. Examen: AGY. |
| BOOT-04 | CONFIRMADO, BAJO | `god_engine.rs:1445-1486` | Los bucles de balance y comisiones reintentan para siempre errores permanentes (401, -2014, -2015, -1022). Los lanzadores de simulación ponen `BINANCE_API_KEY=shadow` y el arranque gira sin fin. | Kimi (lease). |
| QSR-CL-07 | CONFIRMADO, BAJO | `quantum-arena/src/config.rs:8-12` | El doc deriva d* con c = 0,25 (daría 0,348), pero el valor 0,6316 es el de c = 0,5. El valor es correcto; el doc engaña a quien lo reutilice. | Fable / AGY (F1-009). |
| MUE-03 | CONFIRMADO, BAJO | `god-engine-core/src/stateful_engine.rs:828-840` | `FirmasMaxEnt::medir` crea 5 `Vec` cada 64 ticks por moneda y su salida no se publica ni se lee (AGENT_SYNC dice lo contrario). | GLM (GLM-175). |
| MUE-04 | CONFIRMADO, BAJO | varios crates | 3 752 líneas de módulos sin uso fuera de sus tests (Feynman instantón, Wilson RG, Marchenko-Pastur, Lyapunov, Bergomi, Malliavin, fase cuántica…). Compilan en cada build. | Autores (AGY, GLM): conectar en sombra o retirar. |
| QSR-CL-05 | PLAUSIBLE, BAJO | `config.rs:1109-1111`; `lib.rs:2163-2176` | `sync_from_registry` descarta `lg22_arl < 10`, contra F1-089e, y la sincronización siguiente pisa el valor de la gobernanza con el de la config. | Sol (F1-089e) + Codex QRA. |
| FEE-06 | PLAUSIBLE, BAJO | `quantum-arena/src/state.rs:341, 385-397` | El carry de funding usa un intervalo fijo de 8 h; en Binance es por símbolo (4 h y 1 h en muchos). En esos símbolos el carry modelado es 1/2 u 1/8 del real. | Fable (F-A1); poller: Kimi. |

Lectura de los fees (lo que más pesa para la meta): la puerta modela
`2·taker + 2·(piso de slippage + latencia)` ≈ 12,6 pb de nocional con las tasas
por defecto (10,0 con el taker observado de la cuenta). Lo que se paga en vivo es
≈ 14 pb (12 con el taker observado), porque ni el spread ni el deslizamiento real
de la IOC entran en la puerta (LC-F0-16, ya registrado). La física y el backtest
cobran el TP como maker, así que el backtest abarata los TP y Kelly aprende de
eso (registrado). No hay 13 USD fijos en la ruta viva: el balance sale de
`totalWalletBalance` en todos los modos.

## 5. Átomo de esta sesión: RSK-01, la tasa de mantenimiento de la guarda de liquidación

**Por qué este y no otro.** Los 15 hallazgos se ordenaron por daño posible y por
dueño. SEL-02 y BOOT-02 (los de mayor alcance) ya tienen arreglo en el WIP local
de Kimi (B1+B3): duplicarlo en `main` le rompería su rebase. Los de `lib.rs` del
núcleo son del lease de Codex QRA. RSK-01 está en `risk-engine/src/lib.rs` (sin
lease), protege el capital de una pérdida que el modelo de riesgo da por
imposible (liquidación antes del stop) y se puede cerrar con un cambio pequeño
que sólo endurece.

**Qué cambia.** `risk_engine::tasa_mantenimiento_efectiva(medida, L_max)`:

- con una tasa medida para la moneda (`c{id}:maintenance_margin_rate`, finita y
  > 0), esa, acotada a [0,1 %; 5 %];
- sin dato, `max(1 %, 0,5 / L_max)` en el mismo intervalo (≈ la tasa del primer
  tramo de Binance: 1/(2·L_max));
- con `L_max` no finito o ≤ 1, el techo (5 %).

La guarda F1-088 la lee con `get_for_coin_strict` (sin caer a la clave global) y
con el `max_leverage` del spec del símbolo. Para `L_max ≥ 50` el resultado es el
1 % de antes, así que los fixtures del oráculo (T-1, T-2, T-3 y el golden
registran 50×) no cambian por construcción.

**Medido antes del arreglo (sonda sobre `33dfe1064`, 216 configuraciones).**
Con capital de 50 y 100 USD y sin historial, la sonda mínima sale a L = 20 con
0,255 USD de margen. Con stops de 2,5 %, 3 % y 3,5 % en un símbolo de 20×
(tasa real 2,5 %) se cumple `L · (SL + mmr) ≥ 1`: la liquidación aislada queda
antes del stop. Con 13 USD (L = 6) y con 1 000 o 10 000 USD (L ≤ 4) no pasa.

**RED → GREEN** (`crates/risk-engine/tests/rsk01_mmr_sin_escritor_contract.rs`):

| Test | Antes | Después |
|---|---|---|
| sonda de 20× no se liquida antes del stop (capital 50/100, stops 2,5–3,5 %) | FALLA: L = 20, L·(SL+mmr) = 1,000 | pasa |
| la clave global no afloja la guarda | FALLA: tasa global 0,1 % ⇒ L = 20 | pasa |
| la tasa medida por moneda manda (0,4 % ⇒ L = 20) | pasa | pasa |
| símbolo de 50× sin dato: orden idéntica bit a bit a la del 1 % medido | pasa | pasa |
| tasa sin dato frente a L_max (y monótona en 2..200) | — (API nueva) | pasa |
| tasa medida válida manda; NaN, ±∞, 0 y negativos no cuentan | — (API nueva) | pasa |

El contrato F1-088 existente (4 tests) sigue en verde.

**Lo que este átomo no hace.**

- No publica la tasa real. Eso exige leer `GET /fapi/v1/leverageBracket` en el
  ejecutor o el host, que están en el lease de Kimi (pedido en el buzón).
- No cambia `max_leverage` del spec, que en vivo sigue siendo el literal 20 de
  `get_official_binance_spec` (LC-F1-02). En vivo, por tanto, la tasa sin dato
  queda en 2,5 % para todo símbolo, que es conservador para BTC (real 0,4 %).
- El código nunca fija `marginType`: con margen cruzado (el modo por defecto)
  la liquidación de una posición pequeña depende de todo el equity y no de
  `1/L − mmr`. La guarda supone margen aislado, el caso peor.

En micro (13 USD) el techo de apalancamiento es ≈ 5–6,5×. La guarda sólo muerde
con stops por encima del ~12 %, así que la conducta viva con el capital actual no
cambia.

## 6. Recursos: lo medido y lo que se pide a los agentes

Medido el 2026-11-01 entre las 14:10 y las 15:40 (Bogotá), con `du
--apparent-size` sobre el checkout principal y `Win32_OperatingSystem`:

| Recurso | Medido | Lectura |
|---|---|---|
| RAM | 23,38 GiB visibles; 0,79 GiB libres al empezar, 3,09 GiB después; 24 GiB de commit libre | La ocupan ~13 apps de agentes (Kimi Code, ChatGPT, Hermes, Edge, 3× Claude, Codex, 2× ZCode, WorkBuddy), no el bot ni las compilaciones. |
| Disco C | 255,9 GB libres | Sin presión. |
| `target/` | 41 GB: `debug` 36, `release` 3,2, `private-native-exes` 2,6, copias `diag-qra-*` ≈ 0,2 | Un solo `target/` compartido, como pide AGENTS.md. Las copias de ejecutables de QRA son evidencia suya. |
| `.agent_tx/` | **34 GB** | Segunda carpeta más grande del proyecto: recibos y snapshots de Codex QRA. |
| `data/` | 24 GB | Tapes: intocable. |
| `.git` | 3,5 GB | — |
| `logs/` | 1,9 GB (de ellos `logs/agent-fabric` 1,1 GB) | — |
| `graphify-out/`, `.claude/`, `models/` | 777 MB, 288 MB, 4,1 MB | — |

Peticiones (ninguna borra nada ajeno; cada dueño decide):

1. **Codex QRA:** `.agent_tx/` ocupa 34 GB. Archivar o comprimir fuera del repo
   los snapshots de olas cerradas cuyos recibos ya están en `docs/audit/`, y
   decir en el buzón qué conserva y por qué.
2. **Operador:** cerrar las apps de agentes que no estén trabajando es lo que
   más RAM libera; con 0,8 GiB libres una compilación de tests puede quedarse
   sin memoria (error 1455).
3. **Todos:** compilar con el `target/` compartido (`CARGO_TARGET_DIR`
   apuntando al del checkout principal, también desde un worktree hermano);
   para tipos, `cargo check --workspace --all-targets`; para un contrato,
   `cargo test -p <crate> --test <archivo>` en vez de la suite entera del crate.
4. **Camino caliente:** MUE-01 (homología topológica y RedTeam en cada tick sin
   lector) y MUE-03 (5 `Vec` cada 64 ticks sin lector) son CPU y asignaciones
   gastadas en el hilo de decisión. Desconectarlos o pasarlos a sombra
   muestreada es de sus dueños (Codex QRA y GLM).
