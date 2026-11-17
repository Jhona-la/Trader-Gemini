# Qué haría falta en el HOST para juzgar la Demo (DECLARADO, NO implementado)

## Estado tras CABLEADO HOST v7 (17-nov-2026; manda sobre lo histórico de abajo)

Implementado en `src/bin/god_engine.rs` (commits CABLEADOV7-P1..P4 de `integrador/capital-v4l`):

* **Income (§2 abajo):** `escribir_income_jsonl` exporta todas las filas de cada pasada del
  FEE-BREAKER, antes de su partición, a `outputs/income_demo.jsonl` (rotación diaria y
  publicación atómica). No acredita cobertura del exchange ni conciliación.
* **Embudo (§3):** `IMMUNE_SYSTEM` suma las posiciones cerradas de cada intento del aplanado
  inmune (segundo componente de `Ok`, clave agregada `AGREGADO_APLANADO_INMUNE`); `FLATTEN`
  suma una vez cada cierre agrupado `(símbolo, t_ms)` del FEE-BREAKER. Al apagar se vuelca
  `outputs/funnel_demo.json` para `demo_report --funnel`. `record_latency_blocked` sigue sin
  llamador.
* **Contadores (§4):** `bootstrap_orders_sent` (por entrada dimensionada en bootstrap) y los
  cierres `BRACKET_EXCHANGE` contabilizados. El resto sigue SIN CABLEAR (ver
  `PUNTOS_DE_INCREMENTO` en `demo_counters.rs`): los cierres del núcleo y del inmune no tienen
  hoy un neto por cierre acreditado en el host.

Pendiente:

* **`by=CORE` en `DEMO_TEL event=TRADE_CLOSE`:** no se aplicó. El esquema DEMO_TEL de
  `report_demo_run.rs` (`esquema_tel`/`validar_tel`) rechaza un campo fuera de su lista
  («texto fuera de esquema»), así que añadir `by=CORE` solo en el núcleo dejaría cada cierre
  del núcleo como diagnóstico del informe oficial. Necesita una decisión de alcance que incluya
  ese lector.
* **Líneas DEMO con `trade_id` (§1):** los FILL DEMO_TEL ya traen el `trade_id` nativo; el resto
  del §1 histórico sigue como estaba.

Sol M1 · 12-nov-2026 10:xx. Área: `crates/telemetry-server` únicamente.
**Este archivo es una declaración**: no toca `src/bin/god_engine.rs`, ni
`god-engine-core`, ni `execution-engine`, ni las ventanas del motor vivo, ni
`models/`, `config_dir/`, `data/demo/`. No lee `.env` ni llama al exchange.

## 0. Diagnóstico verificado en `logs/corrida_demo_20261112_094131.log` (sólo lectura)

* El motor **no emite** líneas `DEMO SIGNAL/ORDER/FILL/CLOSE`. Sin `trade_id`
  no se puede unir el log con el income del exchange.
* El banner `[ts=…] 🔥 [MODO WARMUP/DEMO ACTIVO] …` contiene `DEMO` y abortaba
  `parse_run_log` con código 2. Corregido en `report_demo_run.rs`: una línea
  sólo es DEMO si, tras `[ts=…]`, su **primer token** es `DEMO`.
* **No hay productor de `income.jsonl`**: bruto, comisión, funding y neto
  conciliados son `N/D`. El `[BRACKET CLOSE]` sí trae `bruto|fees|neto`, pero
  es dinero del host, no conciliado.
* `FunnelCounter` **no está cableado** en el host: los totales del embudo se
  leen de las líneas `[EMBUDO] 60 s …` del log (ver `report_demo_real.rs`).
* Varias líneas salen por `println!` **sin `[ts=…]`** (p. ej. `[FILL]`). El
  segundo lector les presta la marca de la línea con hora más cercana y lo
  declara en la salida; **no fabrica tiempos ni latencias**.

## 1. Líneas DEMO con `trade_id`

1. **Generar el id una sola vez por intención**, en el punto de creación de la
   orden (`src/bin/god_engine.rs`, junto a `new_entry_client_id` y al literal
   `EntryRequest`). Debe ser estable y viajar con la reserva, el reintento, el
   remanente y el cierre. No derivarlo del `client_order_id` si éste cambia en
   un reintento.
2. **Emitir con `telemetry_log!`** (no `println!`) para que `TG_LOG_TS=1`
   anteponga `[ts=…]`. Formato, con `DEMO` como **primer token tras el
   prefijo** y **sin emoji delante** (el parser nuevo es estricto):

   ```text
   DEMO SIGNAL symbol=VETUSDT side=SHORT trade_id=<id>
   DEMO ORDER  symbol=VETUSDT side=SHORT trade_id=<id>
   DEMO FILL   symbol=VETUSDT trade_id=<id> price=0.00512
   DEMO CLOSE  symbol=VETUSDT trade_id=<id> price=0.00510 reason=TP by=BRACKET
   ```

3. **Puntos de emisión sugeridos** (lectura del código, sin editarlo):
   * señal/orden: `src/bin/god_engine.rs` (~5935-5976, creación del `EntryRequest`);
   * fill: `crates/execution-engine/src/user_data_stream.rs:506` (ya tiene
     `client_order_id`, `acum` y `status`);
   * cierre: `src/bin/god_engine.rs:4985` (`[BRACKET CLOSE]`) y
     `crates/god-engine-core/src/lib.rs:5462` (`[CLOSE TRACE]`, ya trae
     `reason`, `gross_pnl`, `exit`, `entry`, `ts`).
4. `reason` y `by` deben rellenarse desde quien decide el cierre, no desde el
   lector; hoy `who_ordered` se reporta `N/D`.

## 2. Productor de `income.jsonl`

* Un **escritor** que vuelque una fila JSON por evento de income, con el
  `trade_id` del apartado 1 y los campos que ya consume `parse_income_jsonl`:
  `symbol`, `income_type`, `income`, `time_ms`, `trade_id`, `tran_id`, `asset`.
* Fuente: el stream de usuario de Testnet (o la lectura REST de income ya
  existente), **en el host**. El informe de `telemetry-server` SÓLO lee ese
  fichero: no abre `.env`, no guarda credenciales, no hace llamadas de red.
* Destino: ruta fuera de `models/`, `config_dir/` y `data/demo/` (el motor
  recarga modelos/genoma en caliente). Rotación y escritura atómica para no
  leer un fichero a medias; una línea corrupta se ignora por línea (ya es el
  comportamiento de `parse_income_jsonl`).
* Hasta que exista, el informe DEBE seguir mostrando `N/D` en las columnas
  conciliadas, aunque el `[BRACKET CLOSE]` traiga `bruto/fees/neto`.

## 3. Volcado del embudo

* Cablear `record_death` / `record_latency_blocked` / `record_immune_veto`
  desde los lazos del host (líneas exactas en `AGENT_SYNC.md`,
  `SOL-M1-FUNNEL-CABLEADO`) y volcar `snapshot_by_gate()` a JSON para
  `--funnel`.
* Mientras no esté cableado, `report_demo_real.rs::funnel_totals_from_log`
  suma las ventanas `[EMBUDO] 60 s` del log **sin** contar la sección
  `acumulado:` (duplicaría) ni la población «con posición abierta».
* Las líneas `[EMBUDO]` del log están truncadas por el escritor (~482 B): el
  lector tolera la cola cortada y el informe declara cuántas lo están.

## 4. Contadores de la sección C de `qoder_contraste_inventario.md`

Especificación: `AtomicU64`/`AtomicI64` publicados al registry (visibles en
telemetría) **sin cambiar ninguna decisión**. La implementación de este
encargo vive en `crates/telemetry-server/src/demo_counters.rs` (banco de
contadores + `snapshot()` + render), porque `solo crates/telemetry-server` y
`sin tocar el motor` prohíben editar los puntos de incremento. Los **sitios
exactos de incremento** quedan declarados en `PUNTOS_DE_INCREMENTO` de ese
módulo, para que el dueño del host los cablee cuando se autorice:

| Contador | Sitio del incremento (declarado, NO editado) |
| --- | --- |
| `host_entry_vetos[9]` | `god_engine.rs` 9 motivos de la lista corregida de `qoder_veto_host.md` §B |
| `bootstrap_orders_sent` / `…_envolvente_vetaba` | `god_engine.rs:5552-5556` (guardar `operable` antes de la rama) |
| `exit_reason_counts[11]` | core `lib.rs:5418-5443` (8), `executor.rs:1008` (IMMUNE_FLATTEN), `reconciliation.rs`, `trade_accounting.rs:176` (brackets) |
| `exit_pnl_net_by_reason[11]` | ledger M1 `trade_ledger.rs` al asentar cada cierre, en centavos |
| `ml_shadow_vetos_aprobados` | sombra `lib.rs:10326-10339`; lector de `red_team_veto_*` / `conformal_gauge_veto_*` |
| `funnel_veto[d][stage]` | `direction_diag.rs:227-245`, quitando la condición de libro plano |

Criterio de Qoder: **medir primero, evolucionar después**; ningún contador
puede alterar un veto, un tamaño, un precio ni un cierre.
