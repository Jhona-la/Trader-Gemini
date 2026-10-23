# QRA F2 — revisión documental del endpoint User Data Stream

Consulta iniciada: 2026-10-23 14:33:58 UTC. Revisión de fuentes públicas y código
fijado por SHA; sin API autenticada, conexiones WebSocket, operaciones, Cargo,
modificaciones de código/CI/índice ni escritura del JSON de evidencia.

**Hallazgo:** el endpoint privado de Producción construido por `main@68839f968`
no coincide con el contrato oficial actual. La retirada de las rutas heredadas
está anunciada para una fecha ya pasada. Esto demuestra una brecha documental
y de configuración que requiere un cambio separado; no demuestra por sí solo
qué respuesta entrega hoy un servidor a nuestras credenciales.

## Fuentes primarias y fechas

- [User Data Streams — Connect](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/user-data-streams#connect)
  documenta la base `wss://fstream.binance.com/private` y el acceso
  `/ws/<listenKey>`; su ejemplo completo usa
  `wss://fstream.binance.com/private/ws/<listenKey>`. La fecha de inicio de la migración se contrasta con el changelog siguiente.
- [Change Log](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/change-log):
  la entrada **2026-03-05** introduce el aviso y las categorías `/public`,
  `/market` y `/private`; **2026-04-02** añade la fecha de retirada
  **2026-04-23**. No es un cambio anunciado por primera vez hoy.
- [Anuncio oficial del 2026-03-06](https://www.binance.com/en/support/announcement/detail/ebf9b0aa9eca4ff3804eef6fb09ba32a)
  confirma disponibilidad inmediata de las nuevas bases y retirada permanente
  de las heredadas el **23 de abril de 2026**.
- [Important WebSocket Change Notice — Compatibility & Migration Guidance](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Important-WebSocket-Change-Notice#compatibility--migration-guidance)
  reitera esa fecha y advierte que conexiones no migradas sólo podrán recibir
  tráfico de la categoría pública; `/market` y `/private` dejan de publicar en
  ellas. También conserva un ejemplo de profundidad pública funcionando en
  una ruta antigua, mientras otra sección anuncia la retirada general. Por
  tanto, no convertir el aviso en la afirmación de que todo handshake o toda
  ruta antigua necesariamente falla. Para UDS, la indicación de migrar a
  `/private` es explícita.

El contrato UDS basado en ruta y los ejemplos privados con parámetros
`listenKey`/`events` del aviso son dos formas documentadas de acceso; no hay que
sustituir una por otra ni limitar eventos accidentalmente para corregir la base.

## Código observado, sin modificarlo

Archivo: `crates/execution-engine/src/user_data_stream.rs`.

| Revisión fijada | Blob del archivo | Ruta construida por `start`, líneas 114–118 y 173 |
| --- | --- | --- |
| `68839f9683afdec64e4c992fdc6ce7e9d7c5441c` | `dab6d270e36c2721595fc8185b0fefadc802d3f9` | Producción: `wss://fstream.binance.com/ws/<listenKey>`; Testnet: `wss://stream.binancefuture.com/ws/<listenKey>` |
| Rama AGY `agy/fase2-capital-observado-y-ffi-safety`, corte `235807dfc512a3654d8037aa80bdc5d19fffdda6` | `180cde569ddd172a21007a0ffbc01f4afe9dcb9e` | Las dos rutas permanecen iguales. El diff observado modifica el tratamiento del balance y sus pruebas, no estos endpoints. |

La selección está en `start` (109–118); `format!` concatena la clave en 173 y
`connect_async` consume la URL en 183. No se infiere que la rama conserve este
estado después del SHA citado ni se certifica aquí su CI.

## Demo/Testnet: alcance distinto

[General Info — Testnet API Information](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info#testnet-api-information)
publica actualmente `https://demo-fapi.binance.com` para REST y
`wss://demo-fstream.binance.com` como base WebSocket. El changelog conserva
además una actualización histórica del **2024-01-24** a
`wss://fstream.binancefuture.com`, distinta del `stream.binancefuture.com` que
aparece en el código. La configuración observada tampoco coincide con la base
Demo publicada hoy.

No se encontró en estas fuentes una fecha de retirada del host Testnet antiguo
ni una especificación explícita de que el prefijo privado de Producción deba
copiarse literalmente a Demo. La fecha de Producción no prueba la retirada de
Demo/Testnet. Tampoco probar Demo bastaría para acreditar la ruta de Producción.
REST, emisión/renovación de la clave y WebSocket deben pertenecer al mismo
entorno; su interoperabilidad actual no se probó en esta revisión.

## Propuesta F2 separada y comprobable

1. Extraer la construcción de URL que utiliza realmente `start` a una función
   pequeña con selección explícita de entorno. Para Producción, adoptar la
   ruta UDS documentada `/private/ws/<listenKey>`; resolver Demo con su contrato
   propio y mantener pareados sus endpoints REST/WS. No introducir fallback
   automático entre entornos o hacia la ruta retirada.
2. Probar esa misma función sin red: esquema/host/path por entorno, una sola
   inserción de `/private` en Producción, clave no vacía y ausencia de mezcla de
   hosts. Si se eligiera la forma con `events`, demostrar que conserva todos
   los tipos de evento que consume el router; el acceso por ruta evita ese
   cambio adicional.
3. Validar después el ciclo operativo de Demo —clave, conexión, renovación,
   reconexión y evidencia de eventos de cuenta— sin publicar claves en logs.
   Distinguir conexión de transporte, vigencia de sesión y recepción de datos;
   no acreditar estado de cuenta conocido únicamente por un handshake.
4. Someter ese cambio a revisión y CI propios antes de integrarlo. Esta nota no
   modifica el candidato/main ya probados ni presenta sus tests offline como
   una certificación operativa de UDS.

No se observaron respuestas HTTP/WebSocket de los endpoints, recepción de
fills, latencia ni una caída real de UDS. Las fechas y contratos anteriores son
afirmaciones de Binance consultadas hoy; el impacto operativo concreto queda
pendiente de una prueba del entorno correspondiente.
