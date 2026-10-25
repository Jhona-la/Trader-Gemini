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
| I-1 | Auditoría del camino socket → cola → métrica → consumidores | La métrica mezcla los cuatro términos | 3 lectores; 18 hallazgos verificados (8 confirmados, 10 plausibles, 0 refutados) y diseño en `docs/audit/linea_i/i1_verificacion_y_diseno.json` | **hecho** |
| I-1b | Rutas `/public` y `/market` (un lector WS por categoría) | En producción la URL heredada no recibe velas ni liquidaciones | Sondas (`SONDAS_WS_2026-10-24.md`) + contrato `data-ingest/tests/rutas_feed_contract.rs` que falla en main | **en main (PR #134, a6472e330)** |
| I-2a | Sello de recepción (`MensajeWs { datos, recibido: Instant }`) transportado por la cola | Hoy la cola lleva bytes sin sello: nadie puede separar red de espera local | Test: el sello sobrevive a la cola y el descarte sigue reconociendo el centinela | **en main (PR #136, 0d0b07e01)** |
| I-2b | Métrica descompuesta `EdadEvento { total, red_reloj, espera_local }` publicada en el arena y en cada aviso | Un evento que esperó 5 s en cola cuenta hoy como red de 5 s | Test puro: E=1000, ahora=6150, espera 5 s ⇒ red 150, local 5000; deriva D del reloj de pared | **en main (PR #136, 0d0b07e01)** |
| I-2c | Contrapresión observable (descartes por periodo, profundidad, espera máxima) por `telemetry_log!` | El monitor sólo imprime con la cola llena y por `println!` | Tests de las ventanas por periodo + contrato de fuente | **en main (PR #136, 0d0b07e01)** |
| I-3a | El inmune cuenta strikes con red+reloj, no con la espera local | Un atraso local aplana posiciones como si la red estuviera caída | `ciclo_strikes` puro; rejilla: nunca suma ni reinicia donde CERT-M4-H04 no lo hacía | **en main (PR #141, 1b8382d20)** |
| I-3b | El rearme del inmune reinicia los strikes de latencia | Tras rearmar, los strikes viejos vuelven a disparar | Contrato de fuente | **en main (PR #141, 1b8382d20)** |
| I-3c | Recorte del bosque sombra con atraso local | El bosque sombra corre en el mismo hilo aunque la cola se atrase | `toca_sombra` puro | pendiente (condicional a I-2) |
| I-4 | Umbral de edad sin deriva evolutiva | El umbral es un gen que la cosecha muta en [500, 10 000] ms sin presión selectiva | 1000 mutaciones lo dejan en 1000·π; aplicar un genoma viejo no lo mueve | pendiente (cambia conducta) |
| I-5 | Reloj coherente: offset del feed medido contra la misma proyección monótona | Offset contra `SystemTime` aplicado sobre `Instant` fijado al arranque | Simulación de una corrección de w32time de −2 s | pendiente (cambia conducta) |
| I-6 | El lector conecta cuando el consumidor está listo | Cada arranque empieza con miles de eventos viejos en cola | `arranque_feed_contract.rs`: el emisor sólo se crea y se suelta tras la fase 6; cada lector espera antes de conectar | **hecho en rama** (cambia conducta del arranque) |
| I-6b | Timeout de conexión y de handshake en `bucle_lector_ws`; stall si no hay primera conexión en `watchdog_secs` | `TcpStream::connect` y `client_async_tls` no tienen timeout: un TLS colgado deja el feed mudo sin stall | Test de la política de espera | pendiente (cambia conducta) |
| I-7 | Sin E/S del almacén de genomas en el hilo caliente | `promote`/`load_active` con fsync cada 5000 mensajes | Contrato de fuente | pendiente (condicional a I-2) |

Ningún átomo toca geometría de órdenes ni dimensionamiento. Los que cambien conducta de
trading llevan T-1 antes de entrar a `main`.

## 4. Recursos

- RAM libre medida a las 13:20: 1,05 GB de 23,4 (otro agente compilando). Sin Cargo local
  mientras la RAM libre siga por debajo de 2 GB; certificación por CI en ese caso.
- Disco C: 343 GB libres. Worktree propio sólo con fuentes (sin `target/` privado).

## 5. Bitácora

- 2026-10-24 ~14:00 — Sondas WS públicas: en producción la URL del host no recibe
  `@kline_1m` ni `!forceOrder@arr` (en demo sí). Red medida: p50 ≈ 150 ms, p99 ≤ 420 ms.
  Evidencia en `docs/audit/linea_i/SONDAS_WS_2026-10-24.md`; aviso en el buzón.
- 2026-10-24 ~14:40 — I-1b implementado en la rama: `data_ingest::rutas_feed`, dos lectores
  (público anclado y crítico; mercado sin anuncio de reconexión ni marca de stall) y la salud
  del feed escrita sólo por eventos públicos.
- 2026-10-24 ~14:45 — I-1b verificado por el semáforo (`-j1`, target compartido):
  `cargo test -p data-ingest` GREEN (12 unitarios, 7 de liquidaciones, 4 del contrato).
  RED: con el `god_engine.rs` de `main` fallan en aserción los tres tests que leen el host
  (`rutas_feed_contract.rs:20`, `:36`, `:58`); el de propiedades pasa. Commit `48dc9ffa0`,
  integrado con `main` 4fb8ca8dd sin conflictos (trae CL-156 en el host).
- 2026-10-24 ~15:30 — I-1b en `main` (PR #134, 10/10 checks). I-1 cerrado: 18 hallazgos
  verificados, 0 refutados; diseño de I-2..I-7 en `docs/audit/linea_i/i1_verificacion_y_diseno.json`.
- 2026-10-24 ~16:00 — I-2 (I-2a/b/c) en la rama `claude/linea-i-i2`. Revisión adversarial antes
  del commit (4 lentes, 2 escépticos por hallazgo): 13 hallazgos, ninguno refutado. Todos
  incorporados: contrato del lado consumidor (`edad_feed_contract.rs`), deriva D del reloj de
  pared en el arena y en los avisos, doc correcta del red+reloj negativo (error de reloj, no
  «reloj adelantado»), `fetch_max` sólo con máximo nuevo y en su propia línea de caché.
  GREEN: `data-ingest` 23+4+7+4, `check --bin god_engine`, `ingesta_ws_host_contract` 7/7.
  RED: con el host de `main` fallan los 4 tests de `edad_feed_contract.rs`.
- 2026-10-24 ~16:20 — I-2 en `main` (PR #136, 10/10 checks, merge 0d0b07e01).
- 2026-10-24 ~16:25 — I-3a/b en la rama `claude/linea-i-i3`. Revisión adversarial del diff
  (4 lentes, 2 escépticos): 9 hallazgos sobreviven (2 MED, 7 LOW), todos incorporados. El
  MED de capital: la primera versión sólo tomaba como datos viejos la cola por encima del
  umbral, y red 2,5 s + cola 700 ms reiniciaba los strikes donde el inmune viejo sumaba;
  ahora el criterio es la edad total. El MED de frescura: el feed quieto sin stall sumaba
  por la banda media antes del watchdog; ahora sólo conserva. Los avisos de «conservados»
  sólo salen cuando se conserva. T-1 no se corre: no ejecuta el host, y lo único que
  comparte con I-3 es un atómico nuevo del arena que el núcleo no lee. Abierto (LOW): los
  strikes conservados no caducan mientras el feed siga degradado.
- 2026-10-25 ~03:00 UTC — I-3 en `main` (PR #141, 1b8382d20). I-6 en la rama `claude/linea-i-i6`.
  Revisión adversarial del parche antes de aplicarlo (3 lentes, 2 escépticos): ningún defecto;
  5 hallazgos (1 LOW, 4 INFO). El LOW: el contrato no impedía que un clon mantuviera vivo el
  emisor y dejara a los lectores esperando para siempre; ahora fija que el emisor sólo aparece
  al crear el canal y al soltarlo. Riesgo aceptado: el arranque queda equivalente a una
  reconexión (unos 0,5–1,5 s sin eventos tras la inicialización, sin el periodo de gracia
  accidental de antes) y los fallos de conexión van en serie tras la inicialización; acotarlos
  es I-6b.
- 2026-10-25 ~02:00 UTC — Sonda de RTT a `GET /fapi/v1/time` desde la máquina del operador
  (HttpClient con conexión reutilizada, 40 muestras por host, la primera descartada por el
  handshake; `docs/audit/linea_i/rtt_probe.ps1`; el CSV de las muestras queda local porque
  `*.csv` está en `.gitignore`, y el script lo regenera).
  Producción: mín 218,5, p50 231,3, p90 267,6, máx 349,8 ms; el 100 % bajo el corte de 350 ms
  del Kalman de `ntp.rs`. Testnet: p50 228,3, p90 299,7, máx 477,1 ms; el 95 % bajo el corte.
  El reloj local va ≈ 38,5 ms adelantado. Responde la pregunta abierta de I-5: el sincronizador
  no queda congelado por el rechazo de outliers. El suelo de ≈ 220 ms es la distancia física
  (Bogotá–Tokio); ningún DNS lo baja, sólo un host más cercano al exchange.

- 2026-10-24 13:25 — Línea abierta; heartbeat en `AGENT_SYNC.md` (commit `2831cf5bd`).
  Auditoría I-1 lanzada (3 lectores, escépticos, diseñador).
