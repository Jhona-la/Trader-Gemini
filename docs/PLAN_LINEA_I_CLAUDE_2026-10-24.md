# Plan de la Línea I (Claude Opus 5.5) — la realidad del feed y de la ejecución

**Fecha:** 2026-10-24. **Base:** `main = 2831cf5bd`.
**Rama:** `claude/linea-i-latencia` (worktree `../tg-lineai`, fuera del repo; compila
sólo por `scripts/cargo_semaforo.sh` contra el `target/` compartido).
**Plan rector:** `docs/PLAN_MAESTRO_FASES_2026-10-22.md` (Sol). Esta línea no abre otro
plan maestro: es un carril de FASE 2 (red, reloj, OMS) con átomos numerados `I-n`.

## 1. Por qué esta línea

Toda decisión del motor (entrar, saltar un scalp, activar el kill-switch, elegir MARKET
directo en el router, el slippage por latencia del riesgo) depende de dos números: cuán
fresco es el dato que se está leyendo y cuánto tarda una orden en llegar al exchange.
El operador reporta más de 200 avisos de «latencia de red elevada». Si esa métrica
mezcla magnitudes distintas, el motor toma decisiones de red sobre un problema que no es
de red, y ninguna mejora de modelos lo compensa.

Ninguna línea activa cubre este carril a 2026-10-24: Línea C retira el censo C-F7b,
Línea G revisa desde la base 517 commits (sólo lectura), Línea H mide el T-3, AGY está en
Fase 7 (ergodicidad, rough paths), GLM en entrelazamiento/MPS, Sol en F1-101, Codex QRA
en F2 (capital, sesión, caché).

## 2. Qué mide hoy el host (verificado en código)

`src/bin/god_engine.rs` calcula `latency_ms = now_ms − E` en el bucle consumidor, donde
`E` es el sello del evento de Binance y
`now_ms = epoch_baseline_ms + (offset_actual − offset_inicial) + instant_baseline.elapsed()`.
El lector WS (`god_engine.rs` ~5716-5750) encola los bytes en `data_ingest::cola_ws`
(acotada a 5 000, descarta el más antiguo) **sin sello de recepción**, y el bucle los
consume después. Por lo tanto `latency_ms` suma cuatro términos que el código no separa:

| Término | Origen | Quién lo puede arreglar |
|---|---|---|
| red + TLS | trayecto exchange → socket | endpoint, DNS, TCP_NODELAY |
| lote del exchange | `@depth5` (250 ms), `@kline_1m` (250 ms) empujan con cadencia | nadie: es del stream |
| espera en cola local | el bucle no da abasto con la tasa de llegada | coste por evento, suscripción |
| error de reloj | offset NTP y proyección monotónica | el reloj del host |

Suscripción viva por símbolo: `@trade` (cada trade individual), `@depth5`, `@kline_1m`,
para todo el universo (26–30 símbolos).

## 3. Átomos

| # | Átomo | Hipótesis | Aceptación | Estado |
|---|---|---|---|---|
| I-1 | Auditoría del camino socket → cola → métrica → consumidores | La métrica mezcla los cuatro términos | Hallazgos con ruta:línea, verificados por escéptico | en curso |
| I-2 | Sello de recepción en el lector WS y métrica descompuesta | Separar red+reloj (`recv − E`) de espera local (`proc − recv`) | Test que falla antes: un evento que espera en cola no cuenta como red lenta | pendiente |
| I-3 | Cada consumidor usa su componente | Kill-switch y avisos de red sobre red; atraso local sobre la cola | Contratos por consumidor | pendiente |
| I-4 | Umbral derivado de la distribución medida | Hoy el umbral es un gen aleatorio (500–10 000 ms) | Cuantil medido de la componente de red, con piso físico | pendiente |
| I-5 | Reloj consistente | Ver hallazgos de I-1 | Test de deriva simulada | pendiente |

Ningún átomo toca geometría de órdenes ni dimensionamiento. Los que cambien conducta de
trading llevan T-1 antes de entrar a `main`.

## 4. Recursos

- RAM libre medida a las 13:20: 1,05 GB de 23,4 (otro agente compilando). Sin Cargo local
  mientras la RAM libre siga por debajo de 2 GB; certificación por CI en ese caso.
- Disco C: 343 GB libres. Worktree propio sólo con fuentes (sin `target/` privado).

## 5. Bitácora

- 2026-10-24 13:25 — Línea abierta; heartbeat en `AGENT_SYNC.md` (commit `2831cf5bd`).
  Auditoría I-1 lanzada (3 lectores, escépticos, diseñador).
