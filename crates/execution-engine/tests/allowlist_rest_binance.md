# Lista blanca de accesos directos a Binance fuera de la puerta REST (PUERTA-REST P5)

La lee `tests/no_reqwest_a_binance_fuera_de_la_puerta.rs`. Cada fila es un archivo `.rs` del workspace
(`crates/`, `src/`, `tests/`, `benches/`, `examples/`) que hoy escribe un host de Binance
(`fapi.binance.com`, `testnet.binancefuture.com`, `fstream.binance.com`, `stream.binancefuture.com`,
`data.binance.vision`) en lugar de pasar por `execution-engine/src/binance_gateway.rs`.

Reglas (las comprueba la prueba):

1. Un archivo que escribe un host y no está aquí hace fallar la prueba: el acceso nuevo va por la puerta.
2. Una fila cuyo archivo ya no escribe ningún host hace fallar la prueba: hay que borrarla (la lista solo se achica).
3. La lista no puede crecer por encima del techo del inventario inicial (`TECHO_LISTA_BLANCA` en la prueba).
4. Cada fila lleva su paso y su deuda. Cada paso P6-P10 trae en su recibo el diff de esta lista.

Inventario inicial: `rg -l` de los cinco hosts sobre esas carpetas en el commit de P4 (`fe3c23405`), 21 archivos.
Tipo: REST = tráfico REST de producción a migrar; WS = WebSocket (la puerta REST no lo cubre); BULK =
`data.binance.vision` (archivo de descargas, fuera del peso de `fapi`); PRUEBA = el host solo aparece como texto en
pruebas o comentarios, sin tráfico; RED-PRUEBA = prueba que habla con testnet de verdad.

| Ruta | Hosts | Tipo | Paso | Deuda |
| --- | --- | --- | --- | --- |
| `crates/data-pipeline/src/omni_multiplexer.rs` | fapi, testnet | REST | P6 | Llamadas periodicas con su propio reqwest: openInterest, premiumIndex, ticker/price y ratios L/S de /futures/data/*; ninguna firma esta en el mapa s1. |
| `crates/data-pipeline/src/historical.rs` | fapi, testnet | REST | P7 | klines y aggTrades con URL escrita a mano; pesos N/D en el mapa (fallan cerrados hasta verificarlos). |
| `crates/data-pipeline/src/dynamic_ranker.rs` | fapi, testnet | REST | P7 | Ranking de simbolos por REST propio: exchangeInfo (N/D) y ticker/24hr (fuera del mapa s1). |
| `crates/data-pipeline/src/api_client.rs` | fapi, testnet | REST | P7 | balance y commissionRate con su propio cliente (pesos verificados en el mapa). |
| `src/bin/god_engine.rs` | fapi, testnet, fstream, stream | REST + WS | P8 | Sondeo de OI del host por REST propio (filas N29-N31); ademas elige el host WS (fuera de la puerta REST). |
| `crates/god-engine-core/src/bootloader.rs` | fapi, testnet, stream | REST | P8 | Velas del calentamiento (klines, peso N/D) con cliente propio. |
| `src/bin/download_history/market_context.rs` | fapi, testnet | REST | P8 | fundingRate de 40 simbolos (fila N47); firma fuera del mapa s1. La v3 la citaba como `src/bin/market_context.rs`. |
| `src/bin/oi_history_sync.rs` | fapi | REST | P8 | openInterestHist siempre en mainnet con curl, sin mirar el estado HTTP (fila N47); firma fuera del mapa s1. |
| `crates/execution-engine/src/client.rs` | fapi, testnet | REST | P9 | Las constantes `BINANCE_BASE_URL` / `BINANCE_TESTNET_URL` y el transporte del ejecutor; P9 delega el HTTP en la puerta. |
| `crates/execution-engine/src/dynamic_symbols.rs` | fapi, testnet | REST | sin asignar (propuesto P9) | Universo dinamico por REST propio (ticker/24hr, fuera del mapa s1). No estaba en el inventario de la v3. |
| `crates/quantum-arena/src/symbol_ranker_engine.rs` | fapi, testnet | REST | sin asignar (propuesto P7) | Ranking por REST propio (exchangeInfo y ticker/24hr). No estaba en el inventario de la v3. |
| `crates/execution-engine/tests/testnet_integration.rs` | testnet | RED-PRUEBA | sin asignar | Prueba que llama a testnet de verdad (ticker/price). Decision: pasarla por la puerta o dejarla fuera del alcance. |
| `crates/data-pipeline/src/ws_client.rs` | fstream, stream | WS | fuera de la puerta REST | Hosts del WebSocket de mercado. Decision del coordinador: si el escaneo de P10 cubre WS. |
| `crates/execution-engine/src/user_data_stream.rs` | fstream, stream | WS | fuera de la puerta REST | WebSocket del user-data stream (el listenKey REST va por client.rs). Misma decision. |
| `crates/data-ingest/src/rutas_feed.rs` | fstream, stream | PRUEBA | fuera de la puerta REST | Solo en sus pruebas unitarias (rutas WS). Misma decision. |
| `crates/data-ingest/tests/rutas_feed_contract.rs` | fstream, stream | PRUEBA | fuera de la puerta REST | Texto de la prueba de rutas WS. Misma decision. |
| `crates/god-engine-core/tests/ingesta_ws_host_contract.rs` | fapi, fstream | PRUEBA | fuera de la puerta REST | Anclas de texto del host WS y del bootloader. Misma decision. |
| `crates/execution-engine/src/executor.rs` | fapi, testnet | PRUEBA | P9 | Solo en `#[cfg(test)]` (la prueba CL-216 y una URL de ejemplo). |
| `crates/quantum-arena/src/protection_health.rs` | fapi | PRUEBA | fuera de la puerta REST | Solo en una prueba (texto de un error de red). |
| `src/bin/binance_vision_sync.rs` | data.binance.vision | BULK | fuera de la puerta REST | Descarga de archivos mensuales/diarios. Decision: si data.binance.vision entra en el presupuesto. |
| `src/bin/download_history.rs` | data.binance.vision | BULK | fuera de la puerta REST | Descarga de velas mensuales. Misma decision. |
