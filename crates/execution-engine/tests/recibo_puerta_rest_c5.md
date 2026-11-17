# Recibo C5 de la puerta REST (PUERTA-REST P1-P5, C1-C4 y F1)

Lo verifica `c5_recibo_declara_el_alcance_del_sha` (`binance_gateway_contract.rs`). Si cambia lo que este
recibo declara pendiente, la prueba falla y el recibo se rehace.

## Sujeto

- Base revisada: `f0a9d4f092351657b8cda1cb06686e064caf1b4b` (C1-C4). El dictamen de cierre de CODEX-CLI
  (Partes 2 y 7) fue FAVORABLE CON CAMBIOS: R1, R3, R4 e I10 quedan APTOS; R2 conservaba el hueco R2-F.
- Corrección: commit PUERTA-F1 y este commit PUERTA-F2, sobre la punta de `integrador/capital-v4l`, que ya
  incluye el lector `.tgb`, CACHEV3 e I11.
- El SHA exacto de este recibo es el del commit que lo introduce. Su compuerta y su simulación se emiten para
  ese SHA. El verde declarado de `f0a9d4f09` no se transfiere a este SHA ni a otro.

## Alcance (texto del dictamen de cierre, Parte 7, conservado tal cual)

> P3 entrega el parser de exchangeInfo.rateLimits y el respaldo LG-12. Quedan pendientes la descarga y el refresco seguros hasta verificar el peso de la firma e integrar el llamador. P1-P5 entregan infraestructura y control de excepciones; las migraciones P6-P10 siguen pendientes. El escaner de literales no demuestra exclusividad global del transporte. Esta publicacion no autoriza Mainnet ni acredita imposibilidad global de ordenes en Mainnet. IBKR permanece como infraestructura de solo lectura sin emisor integrado; su primera conexion debe usar el registro historico de ventana y los demas controles. La validacion y la simulacion deben identificar el SHA corregido exacto.

## Alcance anterior que se conserva (recibo C5 de f0a9d4f09)

- **P3 es parser y respaldo.** `LimitesExchange::desde_exchange_info` lee `rateLimits` de un cuerpo que le da
  quien llama y cae al respaldo LG-12 por contador. `GET /fapi/v1/exchangeInfo` sigue N/D en la tabla de pesos
  y la puerta lo rechaza. No se inventó ningún peso.
- **Las migraciones P6-P10 siguen pendientes.** Ningún llamador del motor pasa por `BinanceGateway`. Los 21
  archivos de la lista blanca (`allowlist_rest_binance.md`) siguen accediendo directo; C4 solo impide que la
  lista crezca o se sustituya.
- **No hay garantía global de exclusividad REST.** La puerta es una ruta más; el tráfico heredado de la lista
  blanca sale por fuera. C4 sigue siendo un escaneo de literales por archivo.
- **No hay garantía de que sea imposible emitir órdenes en Mainnet.** El host conserva el armado humano y la
  ejecución no-papel, y OMS conserva `ensure_hedge_mode`. `BinanceGateway::para_red(false, …)` apunta a Mainnet y
  transporta firmas de órdenes sin una guarda propia de Testnet. Hoy no tiene llamadores productivos.

## F1: cierre de R2-F

- `url_de_transporte` arma el `reqwest::Url` que se envía. Antes de la firma, el freno y la reserva, falla
  cerrado (`PUERTA_ALCANCE`) si la URL lleva fragmento o si su consulta no es, byte a byte, la que dio quien
  llama: espacios, comillas, `<`, `>` y bytes no ASCII se codifican; tabuladores, saltos de línea y espacios de
  los extremos se quitan.
- El alcance (con o sin `symbol`) se lee de `url.query()` de ese objeto, y `request` recibe el mismo objeto.
  Los bytes válidos, incluidos los escapes `%` ya hechos, salen sin recodificar.
- Prueba: `f1_fragmento_no_cambia_el_alcance`, con `openOrders` y `openAlgoOrders` y la consulta
  `timestamp=1&signature=FIRMA_FICTICIA#&symbol=BTCUSDT`: rechazo sin envío ni reserva, más los controles de
  consulta válida (1 con `symbol`, 40 sin él) y de las normalizaciones. Fue ROJA sobre el código de
  `f0a9d4f09`: salía `?timestamp=1&signature=FIRMA_FICTICIA` y se cobraba 1.
- Límite: la comparación cubre lo que `url` 2.5.8 normaliza. Que `http` 0.2.12 acepte esa consulta sin
  cambiarla lo prueban los casos válidos del servidor local (la línea recibida es la consulta tal cual); para
  el resto de bytes se apoya en su lista de bytes permitidos (CODEX, evidencia 34).
