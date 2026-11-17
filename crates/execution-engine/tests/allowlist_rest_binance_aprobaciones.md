# Aprobaciones de la lista blanca REST (PUERTA-REST C4)

La lee `tests/no_reqwest_a_binance_fuera_de_la_puerta.rs` junto con `allowlist_rest_binance.md`. Es la
cadena de aprobaciones de la lista blanca, en orden. Reglas (las comprueba la prueba; dictamen de CODEX-CLI sobre
698945c27, R4/C4):

1. A0 es el inventario inicial de P5 (21 archivos). Su huella SHA-256 esta fijada en la prueba (`HUELLA_A0`).
2. Cada aprobacion nueva es SUBCONJUNTO de la anterior: sin altas ni reingresos. Lleva su recibo y la linea
   `Bajas:` con exactamente las rutas que quita (el diff del recibo).
3. La lista blanca vigente es exactamente la ultima aprobacion. Quitar una fila sin su bloque falla; anadir una
   (aunque no pase del techo, o aunque sea una ruta que ya estuvo) falla: el acceso nuevo va por la puerta.

Un paso P6-P10 que migra archivos: borra sus filas de la lista blanca y anade aqui un bloque `## Ai` con su
recibo, sus bajas y la lista que queda.

## A0 · P5 · da829673b
Recibo: PUERTA-REST P5 (da829673b), inventario `rg -l` de los cinco hosts en P4 (fe3c23405); respuesta del integrador 20261116-0347-puerta-rest-p1p5
Bajas: ninguna (inventario inicial)
- `crates/data-ingest/src/rutas_feed.rs`
- `crates/data-ingest/tests/rutas_feed_contract.rs`
- `crates/data-pipeline/src/api_client.rs`
- `crates/data-pipeline/src/dynamic_ranker.rs`
- `crates/data-pipeline/src/historical.rs`
- `crates/data-pipeline/src/omni_multiplexer.rs`
- `crates/data-pipeline/src/ws_client.rs`
- `crates/execution-engine/src/client.rs`
- `crates/execution-engine/src/dynamic_symbols.rs`
- `crates/execution-engine/src/executor.rs`
- `crates/execution-engine/src/user_data_stream.rs`
- `crates/execution-engine/tests/testnet_integration.rs`
- `crates/god-engine-core/src/bootloader.rs`
- `crates/god-engine-core/tests/ingesta_ws_host_contract.rs`
- `crates/quantum-arena/src/protection_health.rs`
- `crates/quantum-arena/src/symbol_ranker_engine.rs`
- `src/bin/binance_vision_sync.rs`
- `src/bin/download_history.rs`
- `src/bin/download_history/market_context.rs`
- `src/bin/god_engine.rs`
- `src/bin/oi_history_sync.rs`
