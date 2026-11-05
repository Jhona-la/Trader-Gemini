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
| I-3c | Recorte del bosque sombra con atraso local | El bosque sombra corre en el mismo hilo aunque la cola se atrase | `toca_sombra` puro | **en main (61e44c858, rescate GLM-174)** |
| I-4 | Umbral de edad sin deriva evolutiva | El umbral es un gen que la cosecha muta en [500, 10 000] ms sin presión selectiva | 1000 mutaciones lo dejan en 1000·π; aplicar un genoma viejo no lo mueve | **en main (07c53757a, rescate GLM-171)** |
| I-5 | Reloj coherente: offset del feed medido contra la misma proyección monótona | Offset contra `SystemTime` aplicado sobre `Instant` fijado al arranque | Simulación de una corrección de w32time de −2 s | **en main (2d06e57d1, rescate GLM-173)** |
| I-6 | El lector conecta cuando el consumidor está listo | Cada arranque empieza con miles de eventos viejos en cola | `arranque_feed_contract.rs`: el emisor sólo se crea y se suelta tras la fase 6; cada lector espera antes de conectar | **en main (b3968b314, certificado por GLM-167)** |
| I-6b | Timeout de conexión y de handshake en `bucle_lector_ws`; stall si no hay primera conexión en `watchdog_secs` | `TcpStream::connect` y `client_async_tls` no tienen timeout: un TLS colgado deja el feed mudo sin stall | Test de la política de espera | **en main (59f1989cc, rescate GLM-168)** |
| I-7 | Sin E/S del almacén de genomas en el hilo caliente | `promote`/`load_active` con fsync cada 5000 mensajes | Contrato de fuente | **en main (bcd94be12, rescate GLM-170)** |
| I-8 | Un bracket que sólo rechaza el freno temporal no escala a X-009 (fila N18 del censo C-F7b) | Con un 429/418 activo, los 3 intentos de bracket de una entrada llenada, el cierre de emergencia y la purga se rechazan en local y X-009 arma el kill-switch PERMANENTE por un freno que el exchange levanta solo (contrario a M4-C02) | `execution_engine::freno_temporal` (clasificador y decisión) recorriendo el ejecutor real frenado; contrato del host `i8_bracket_frenado_contract.rs` que falla en main | **en main (PR #164, d67b90cf2)** |
| I-11 | Las reconexiones WS esperan cada vez más y caben en el límite de Binance (fila N56) | Cada fallo de conexión del lector esperaba 1 s fijo, el único `retry_count += 1` estaba en un brazo inalcanzable y el contador volvía a cero con el handshake (un servidor que acepta y corta, cada 100 ms); el user-data stream volvía a 500 ms al obtener el listenKey y el spot reconectaba sin esperar tras un corte | `quantum_arena::reconexion_ws` (escalera por cliente, reinicio con el primer mensaje que no es un Close, `Retry-After` de un 429/418) en el lector, el user-data stream y el spot; prueba de conducta con un servidor WS local que acepta y corta; contrato `i11_reconexion_ws_contract.rs` con el presupuesto de conexiones | **en main (PR #168, 722e29b5f)** |
| I-9 | La pierna que sobrevive a la purga tras un cierre no queda armada sobre la posición siguiente (fila N15b) | Las purgas que fallan tras un cierre (PURGE OCO, EMERGENCY, NAKED-ESCALATION) sólo se reportaban; nada bloqueaba la reentrada y el vigilante no purga una pierna con posición en su símbolo o lado | `execution_engine::purga_pendiente` (registro por (símbolo, lado), reintento del vigilante sólo sin posición que proteger) y contrato del host `i9_purga_pendiente_contract.rs` que falla en main | **en main (PR #165, a834074c7)** |
| I-10 | El aplanado del sistema inmune que no se completó se reintenta (fila N19) | El inmune quedaba `latched` aunque el aplanado devolviera Err: con el freno 429/418 no salía ningún cierre (N5/RV14) y tras un rechazo del exchange (`FLATTEN_INCOMPLETE`) la posición quedaba abierta, con la protección del vigilante pero sin cierre, y nadie lo reintentaba. Además el aplanado daba por cerrado un cierre llenado en parte y purgaba sus brackets | `execution_engine::aplanado_inmune` (frenado: reintento cada ciclo de 5 s; otro error: espera doble de 5 s hasta la cadencia del vigilante, 60 s, desde el final del intento) recorriendo el ejecutor real frenado; reintentos sin barrido de huérfanas (`flatten_open_positions`); `tras_ack_de_cierre` (llenado parcial a X-026); exclusión del apagado; contrato del host `i10_aplanado_inmune_contract.rs` que falla en main | **en main (PR #167, 031a8ddce)** |
| I-12 | La transición del ejecutor al terminar el calentamiento no se aplaza por la entrada del propio tick ni reconstruye un ejecutor por mensaje (fila N48) | La reserva de entrada del tick es una ranura abierta sin confirmar: con ella `replace_executor` rehusaba siempre (`OMS_EXECUTOR_BUSY`), y el host reintentaba mientras el núcleo siguiera proponiendo entradas, construyendo otro ejecutor (cliente reqwest nuevo), revocando la sesión y encolando otro `refresh` en cada intento: en papel casi en cada mensaje WS, en una sesión real en cada ciclo de refresco | `OmsSession::paso_transicion` (suelta antes la reserva del tick; con la sesión ocupada por otra causa no construye ni revoca; tras un rechazo, a lo sumo un refresco de reintento en curso, `ReintentoTransicion`) y contrato `tests/i12_transicion_oms_contract.rs` (6 pruebas; la guarda del host falla en main) | **en main (PR #169, e3b3481a2)** |
| I-13 | El ejecutor que el host publica al terminar el calentamiento hereda del vigente el freno 429/418, su reloj y el uso del exchange (fila N26; failover de Línea C por `executor.rs` y `client.rs`) | El host construía el sucesor con `new_with_shared`, que estrena freno y uso a cero: dentro del Retry-After de un 429 o del baneo de un 418 el sucesor volvía a enviar, por encima del 80 % de la cuota seguía admitiendo lecturas, y las plazas críticas del anterior en vuelo no contaban | `OrderExecutor::sucesor_de` (comparte el `Arc<FrenoExchange>`, el Arc de `arena` y `Arc<Mutex<UsoExchange>>` con la misma URL, clave y arena, o nada y lo dice: `HerenciaDeCuota`); `tests_n26` en executor.rs (8) y contrato `tests/i13_sucesor_cuota_contract.rs` (4; la guarda del host falla en main) | **rama `claude/linea-i-i13`** |

Ningún átomo toca geometría de órdenes ni dimensionamiento. Los que cambien conducta de
trading llevan T-1 antes de entrar a `main`.

## 4. Recursos

- RAM libre medida a las 13:20: 1,05 GB de 23,4 (otro agente compilando). Sin Cargo local
  mientras la RAM libre siga por debajo de 2 GB; certificación por CI en ese caso.
- Disco C: 343 GB libres. Worktree propio sólo con fuentes (sin `target/` privado).

## 5. Bitácora

- 2026-11-05 ~00:05 UTC — I-13 (= N26) en la rama `claude/linea-i-i13`, por failover: la fila es de Línea C (método en `executor.rs`) y de Línea I o AGY (línea del host), y Línea C lleva más de 48 h sin actividad. Verificado en el código: `new_with_shared` estrena un `FrenoExchange`, un reloj y un `UsoExchange`; el freno y la cuota son de la IP y de la cuenta, no del ejecutor. `sucesor_de` comparte los tres Arc sólo con la misma URL, la misma clave, la misma arena (`Arc::ptr_eq`) y un freno del anterior que decide con esa arena; si no, estrena y el host lo manda a `telemetry_err!`. Las plazas críticas del anterior se sueltan en el mismo candado (su `Drop`), y lo perdido sin respuesta (N24) cuenta en el sucesor; los clones del cliente anterior (user-data stream, NTP) frenan al sucesor y él a ellos. No toma candados; no hereda kill-switch, limitador local ni cachés. RED con el sucesor que no comparte: 6 de 7 fallan por su causa; GREEN 8/8; tres mutantes («no comparte», «sin comprobar el reloj», «sin compartir el uso») detectados; la guarda del host falla con el host de main. Revisión con tres lectores, tres diseños y tres críticos: de ella, la prueba de las cachés llena las del anterior y fija que el kill-switch de la arena sobrevive, la del 418 comprueba el clon del sucesor y la bitácora, y los casos sin herencia comprueban el reloj de la arena pedida. Residuales: con la misma URL y otra clave estrena freno aunque el 418 es de la IP (decisión de Línea C); el kill-switch del ejecutor no cruza el cambio (fila nueva N58); las cachés del sucesor empiezan vacías (un exchangeInfo y un `set_leverage` más por símbolo).
- 2026-11-04 ~18:25 UTC — I-12 en main (PR #169, e3b3481a2): CI 12/12 en verde al fusionar. Retirados la rama `claude/linea-i-i12` (local y remota) y su worktree; sólo quedaban cuatro carpetas vacías que crearon las pruebas locales.
- 2026-11-04 ~15:30 UTC — I-11 en main (PR #168, 722e29b5f): CI 12/12 en verde al fusionar. Retirados la rama `claude/linea-i-i11` (local y remota) y su worktree; los archivos ignorados que quedaban eran restos de las pruebas locales.
- 2026-11-04 ~02:30 UTC — I-12 (= N48) en la rama `claude/linea-i-i12`. Verificado en el código: la reserva de entrada que el núcleo deja en el tick de la transición es una ranura abierta sin confirmar (`cancel_unconfirmed_generation`), y `replace_executor` rehúsa con cualquier ranura así; el host de main cancelaba la reserva DESPUÉS del rechazo, hacía `continue` y lo repetía. Con una señal que persiste la transición no llegaba nunca, y cada intento construía otro `OrderExecutor` (cliente reqwest nuevo), revocaba la sesión (ready, comisiones observadas, publicación) y encolaba otro `refresh` en `refresh_serial`. En papel (demo en mainnet sin MAINNET_ARMED) el refresco no hace REST y vuelve a Ready en el acto, así que el intento llegaba casi en cada mensaje WS; en una sesión real, en cada ciclo de refresco REST. Medido con la sesión OMS real y reservas reales (contrato, prueba 2, caso de papel): con el protocolo de main, 100 mensajes con entrada propuesta dan 100 ejecutores construidos, 100 refrescos encolados y ninguna transición; con `OmsSession::paso_transicion`, 1 ejecutor, 0 refrescos y transición en el primer mensaje. Con la sesión ocupada por otra causa (otra ranura sin confirmar) no se construye ni se revoca, y avisa una sola vez; tras un rechazo, un solo testigo de refresco (`RefrescoDeReintento`) y hasta soltarlo no se construye otro ejecutor. El tick de la transición sigue sin abrir exposición (antes tampoco: la reserva se cancelaba tras el rechazo). RED: el contrato con el host de main (ddc722873) da 5 pasan / 1 falla, la guarda del host; con I-12, 6/6. Revisión adversarial antes del push (4 lentes, 2 escépticos por hallazgo): 6 hallazgos, 0 sobreviven; de ella, la prueba 5 comprueba que la reserva no cancelable deja la protección sucia y que la propia no la ensucia. `host_oms_contract` 19/19 y las pruebas del binario 5/5. Residuales: mientras la transición espera, el mensaje sigue haciendo `continue` (como antes) y no despacha cierres del núcleo en ese mensaje (N48b); el sucesor sigue estrenando freno y uso (N26, con Línea C por `executor.rs`).

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

- 2026-11-03 ~21:30 UTC — I-10 en PR #167 (Auto-fix vigila su CI). I-11 (= N56) en PR #168 (rama `claude/linea-i-i11`): el bucle del lector WS del host esperaba 1 s fijo tras cada fallo de conexión (DNS, TCP, handshake), su único `retry_count += 1` estaba en un brazo inalcanzable (el aviso «unreachable pattern» de rustc) y el contador volvía a cero al completar el handshake, así que un servidor que acepta y corta se reintentaba cada ~100 ms. Revisando los demás clientes WS del proceso apareció el mismo defecto en dos más: el user-data stream volvía a 500 ms al obtener el listenKey (REST) antes de abrir el WS, y el feed spot no esperaba nada tras el corte de una conexión hecha. Ahora los tres usan `quantum_arena::reconexion_ws::EsperaReconexion` (cada uno con su escalera: lector 100 ms·2^n hasta 5 s, user stream 500 ms·2^n hasta 30 s, spot 500 ms·2^n hasta 10 s), que sólo vuelve a cero con el primer mensaje que no es un Close y respeta el `Retry-After` de un 429/418 del handshake. Peor caso en la ventana de Binance (300 conexiones cada 5 min por IP): 2×65 + 15 = 145 de futuros y 34 del spot. RED contra las fuentes de main dfc07271: el contrato `i11_reconexion_ws_contract.rs` da 0/7 y la prueba de conducta del bin (un servidor WS local que acepta y corta) cuenta 30 conexiones del lector en 3 s, una cada ~100 ms; con la rama, 7/7 y a lo sumo 6. En local, sobre la rama: `cargo check --workspace --all-targets --locked` sin errores y sin el aviso «unreachable pattern» del host; `quantum-arena` reconexion_ws 7/7, `execution-engine` reconexion_ws 1/1, `god_engine --bin` 6/6, `data-pipeline` spot_feed 3/3.
- 2026-11-03 ~17:00 UTC — I-10 (= N19) en la rama `claude/linea-i-i10`. Verificado como defecto: el inmune se marcaba hecho (`latched = true`) en los dos brazos del aplanado y su cerrojo era `if latched { continue; }`, así que un aplanado frenado por un 429/418 (que devuelve el freno antes de enviar ningún cierre) o incompleto (`FLATTEN_INCOMPLETE`) no se volvía a intentar; el vigilante re-protege esas posiciones pero no las cierra. Ahora el intento vive en `aplanado_inmune` (el host) y su resultado se anota en `execution_engine::aplanado_inmune::ReintentoAplanado`: con el cerrojo puesto, un aplanado frenado se reintenta en cada ciclo del inmune (5 s; no envía cierres hasta que caduca el freno) y tras otro error la espera se dobla desde un ciclo hasta la cadencia nominal del vigilante (60 s). El rearme por archivo abandona el pendiente. Los contratos CL-44/CL-44b del kill-switch leían el aplanado dentro del bloque del disparo: siguen las mismas aserciones dentro de `aplanado_inmune`; el de I-3b anclaba en `if latched { continue; }`, que ya no existe, y ancla en `if latched {`. Residual N19b (sin verificar): una entrada en vuelo al dispararse el inmune que se llena después de la foto de posiciones del aplanado queda abierta con un aplanado cumplido.
- 2026-11-03 ~19:00 UTC — I-10, revisión adversarial antes del PR (4 lentes, 2 escépticos por hallazgo: 10 confirmados, 4 refutados). Arreglado en la misma rama: (1) la espera contaba desde el inicio del intento y cada reintento repetía el barrido de órdenes huérfanas (todos los símbolos sin posición de positionRisk, dos cancelaciones cada uno): intentos largos podían encadenarse y tener al vigilante sin auditar. Ahora se anota el instante en que el intento TERMINA y los reintentos usan `OrderExecutor::flatten_open_positions` (el mismo aplanado sin barrido; el cuerpo vive en `aplanar_cuenta`, y los contratos de fuente de Línea C que leían `flatten_all_positions` leen ese cuerpo). (2) Un cierre llenado en parte (PARTIALLY_FILLED, o EXPIRED con parte ejecutada) se daba por cerrado y se purgaban los brackets con el resto abierto: `tras_ack_de_cierre` sólo cierra con FILLED o con toda la cantidad ejecutada; lo demás va a X-026. (3) El aplanado del apagado podía coincidir con un reintento: `APAGANDO` / `APLANADO_INMUNE_EN_CURSO` (SeqCst) y espera de hasta 30 s. (4) Contratos con el `.await` fijado y textos corregidos («sin enviar cierres», «cancelaciones de órdenes»; el aviso del apagado decía «cerradas» y «ya planas» con los números cambiados). Medido: las tres pruebas de llenado parcial fallan con la regla anterior (3/11 rojas; la del aplanado sin barrido es API nueva) y pasan con la nueva (11/11); el contrato del host contra el host del commit anterior 2/5 → 5/5. Residual N19c (visto leyendo, sin prueba): el cierre por WS del aplanado y el reenvío de X-026 (`execute_reduce_only_market`) siguen dando por cerrado un llenado parcial.

- 2026-11-03 ~15:30 UTC — I-9 en main (PR #165, a834074c7). La puerta de clippy sobre líneas cambiadas, corrida en local (`RUSTUP_TOOLCHAIN=nightly-2026-06-30 python scripts/puerta_lineas_cambiadas.py --base origin/main --solo clippy`; `--cargo` sólo admite un ejecutable), cazó 5 `cloned_ref_to_slice_refs` en las pruebas nuevas antes de fusionar; corregido (bb05203b6) y en 0. Abiertos para Línea I: N18b, N15c y N19 (aplanado inmune marcado hecho con FLATTEN_INCOMPLETE, sin verificar como defecto).

- 2026-11-03 ~14:00 UTC — I-8 en main (PR #164, d67b90cf2): CI 11/12 en verde al fusionar (replay-contracts en curso) y, en local, execution-engine 460/0 (5 ignorados de testnet), guardas del host 47+8+22+72 y `cargo check --workspace --all-targets --locked` en verde. I-9 (= N15b) en la rama `claude/linea-i-i9`: una purga de brackets que falla tras un cierre (PURGE OCO, EMERGENCY, NAKED-ESCALATION) o la escalada que encuentra la posición ya cerrada dejan la purga pendiente por (símbolo, lado) en `execution_engine::purga_pendiente::PURGAS_PENDIENTES` y despiertan al vigilante; el vigilante la reintenta antes de `purge_orphan_legs`, sólo si no hay posición que esas piernas puedan proteger (mismo lado en hedge, cualquiera del símbolo en one-way), y la retira con un Ok; mientras tanto la decisión de envío veta las entradas en ese símbolo con rollback, como el FEE-BREAKER. Medido: contrato del host contra el host de main 1 pasa / 4 fallan por la causa; con I-9, 4/4 del módulo (reintento sobre el ejecutor real frenado y en papel, sin red), 5/5 del contrato, execution-engine completo y guardas del host en verde. Residual N15c: el cierre X-008 purga en un spawn; una reentrada en el mismo símbolo y lado entre el cierre y su purga puede perder sus brackets nuevos.

- 2026-11-03 ~09:30 UTC — Vuelta tras quedarse esta sesión sin cuota (31 Oct). GLM 167-174 rescató e integró I-6, I-6b, I-3c, I-4, I-5 e I-7 (tabla actualizada con sus commits). I-8 (= N18 de Línea C) en la rama `claude/linea-i-i8`: el host junta los rechazos de los tres intentos de bracket y, si todos son del freno temporal (`RATE_BRAKE`, `RATE_LIMIT_COOLDOWN`, 429/418, cuota del exchange), no cierra ni arma el kill-switch: anota el diario de la posición y deja la protección sucia para el vigilante (CL-44), que la re-bracketea al acabar el freno; durante el freno su lectura de posiciones también frena y no audita. Con un solo rechazo genuino del exchange, X-009 sigue igual. Residual N18b (no tocado): si el bracket lo rechazó el exchange de verdad y el cierre lo bloquea el freno, X-009 arma el kill-switch permanente como antes. Medido: el contrato del host contra el host de main da 1 pasa / 2 fallan por la causa (no juntaba los rechazos; no había rama diferida); con I-8, 6/6 del módulo (incluida la cadena X-009 sobre el ejecutor real con el freno 429 y 418 armado, sin red) y 3/3 del contrato; `cargo check --bin god_engine` en verde. T-1 no aplica: sólo cambia el host.

- 2026-10-24 13:25 — Línea abierta; heartbeat en `AGENT_SYNC.md` (commit `2831cf5bd`).
  Auditoría I-1 lanzada (3 lectores, escépticos, diseñador).
