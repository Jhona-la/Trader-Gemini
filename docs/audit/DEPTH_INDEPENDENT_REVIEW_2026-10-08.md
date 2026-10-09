# Revisión independiente R4-I6 y alcance productor → consumidor

Fecha: 2026-10-08. Revisor: system_inventory. Estado: **ABIERTO**.
Fuente fijada: `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`, árbol
`10b8a427dfbdd5b203595b039b2dad50c69d6f68`. No se ejecutaron Cargo,
rustc, el harness, el runner, red del exchange ni operaciones financieras.
Se leyeron fuentes, recibos y documentación oficial; sólo se recalcularon
hashes/metadatos. Este informe no convierte un archivo runtime en verificado.

## Veredicto sobre el testigo

El mecanismo y las seis aserciones concuerdan con R4-I6. `extract_wall_sum`
busca la secuencia literal `]]` sobre el resto del mensaje, en vez del cierre
estructural del lado. Los dos keys minificados sólo eligen la rama rápida;
no garantizan el formato de los arrays. El espacio entre cierres o un lado
vacío permite alcanzar el lado siguiente. La referencia usa `serde_json`
para validar los seis JSON y sumar cantidades dentro del array correspondiente.

| Entrada | Referencia (bid, ask) | Observación registrada | Interpretación |
|---|---|---|---|
| Un nivel por lado, minificado | (2, 3) | Some((2, 3)) | Control correcto |
| Espacio entre cierres de bids, ambos keys minificados | (2, 3) | Some((5, 3)) | Contaminación de bid con ask |
| Igual espacio interno, key asks espaciado | (2, 3) | Some((2, 3)) | Control del fallback |
| bids vacío y asks con cantidad 3 | (0, 3) | Some((3, 3)) | Copia de ask a bid |
| asks primero y sus cierres espaciados | (2, 3) | Some((2, 5)) | Contaminación simétrica |
| Un nivel 1e308 por lado y cierres bids espaciados | (1e308, 1e308) | None | Overflow sintético entre lados, no dentro de cada lado |

Son seis escenarios de caracterización en un ejecutable. Las aserciones
esperan deliberadamente el comportamiento defectuoso; exit0 **no prueba
corrección**. Los tres comandos registrados son versión, compilación y ejecución,
no tres suites ni seis tests del workspace. La salida `-0` del fold vacío
equivale numéricamente a cero y no cambia el contraejemplo.

El intento anterior `e1ec3a38-4b83-4625-99d5-b0a26f184cde` terminó INCOMPLETE:
la compilación exigía `panic=abort` para fast_float y el wrapper usaba unwind.
No ejecutó el testigo y no es RED. El recibo correcto añade `-C panic=abort`.
No se observó manipulación de fuente ni sustitución de parser por una réplica.

## Identidad comprobada por lectura y hashes

Evidencia: `target/r4-depth-witnesses/02c6b918-d0e2-40a1-abc7-617432fb6c76`.

- Parser completo: 462 líneas, blob `6c47744a94f5bea150e8a65c750b2a2f58c73cc4`;
  SHA256 `2eaf3e07f6ef1ead73d996c3fcff537354ca37b57b39a0ee3fe45e77cf07f5a7`.
  El archivo de evidencia coincide byte por byte con `git show e9:.../parser.rs`.
- Runner: 94 líneas, SHA256 `c0e3302b1660d9349ff12477b58b96a71cb648ab06bd7694d837727810a79a2e`.
- Wrapper: 33 líneas, SHA256 `9bd8a19279d0660e7f0fb21a18249cca854d1fdeb936cbe45eb24e92219cf887`;
  copia ejecutada e input de fixtures idénticos.
- Recibo local completo: SHA256 `57a250ad0a154f498dc2885d0d0e8c883182a8c7ca1ab4ee0360c0e766d3f604`.
- Los tres logs coinciden con sus hashes registrados. Log de ejecución:
  `7290a0979235c69608335df0a562671044f67d577431b97394b5c7a0cd115869`.
- Los rlibs directos memchr, fast_float y serde_json existen y sus hashes
  coinciden con el recibo. El compilador registrado es rustc 1.98.0-nightly,
  commit `096694416a41840709140eb0fd0ca193d1a3e6ba`, MSVC x86_64.

Límite de reproducibilidad: el runner usa dependencias previamente compiladas
y `-L dependency=...`; no inventaría hashes de todas las dependencias transitivas,
del enlazador o del ejecutable. El recibo da identidad de fuente, wrapper y
dependencias directas y una ejecución registrada; no constituye build hermético.
La revisión independiente no repitió esa ejecución.

## Esquema oficial y alcance de las dos rutas

USD-M **Partial Book Depth Streams** incluye niveles 5/10/20 y la suscripción
`@depth10@100ms`; los arrays de respuesta son `b` y `a`. La documentación
actual anuncia endpoints bajo `/public/`; no se probó conectividad ni
compatibilidad del endpoint antiguo del proyecto. [Binance, WebSocket público USD-M](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/public).

En cambio, el endpoint REST `GET /fapi/v1/depth` documenta `bids` y `asks`.
Compartir conceptos de libro no hace intercambiables los dos esquemas.
[Binance, Order Book REST USD-M](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).

1. `BinanceStreamer` construye una URL combinada con bookTicker, aggTrade y
   depth10 (`ws_client.rs:115–120`). El dispatch prueba BookTicker, después
   AggTrade y finalmente DepthEvent (`279–465`). Un evento Depth USD-M normal
   contiene b/a como arrays, no los escalares b/B/a/A del BookTicker ni p/q/m
   del AggTrade. DepthEvent exige `bids` y `asks` antes incluso del fallback
   (`parser.rs:223–227`), por lo que ese payload se descarta con None.
   Envolverlo en `data` no cambia los nombres ni supera el guard.
2. Si a ese método llegase un payload bids/asks aceptado, las sumas se publican
   por `update_l2_depth` (`ws_client.rs:455–464`; `state.rs:754–769`), y los
   campos pueden ser consumidos por el veto del muro (`core/lib.rs:1546–1568`,
   `calibration.rs:327–347`). Esto es una conexión estática condicional.
3. La búsqueda `git grep` de `BinanceStreamer`, `DepthEvent` y
   `start_with_trade_handler` en todos los archivos Rust versionados de e9
   encuentra la implementación, reexport y tests, pero **ninguna instanciación
   productiva de BinanceStreamer**. No se presume uso desde consumidores
   externos al repositorio o carga dinámica que esta búsqueda no observa.
4. `src/bin/god_engine.rs` tiene otra ruta: suscripción depth5 (`873–880`),
   parser `parse_binance_depth5_levels` (`3000–3009`), con b/a estructurales
   (`src/parsers.rs:63–113`). El bin publica las cantidades del mejor nivel
   mediante `update_l2_depth(coin_id, dbq, daq)` en línea 3201. Sus muros
   máximos nocionales para spoofing (`3055–3080`) son un cálculo distinto.
   R4-I6 no está demostrado en ese parser separado ni en una operación suya.

Conclusión de alcance: **R4-I6 es un defecto aislado reproducido y abierto**;
la incongruencia USD-M b/a → parser bids/asks es además un contrato de integración
estáticamente incompatible si se activa BinanceStreamer. No se ha demostrado
que el exchange emitiera los fixtures bids/asks, que esa clase esté activa, ni
incidencia en producción, pérdidas o causalidad sobre el objetivo financiero.
No convertir la ruta condicional de campos compartidos en afirmación de incidencia.

## Lecturas concretas y límites

- `crates/data-pipeline/src/parser.rs`: **1–462, completo**, incluidas utilidades,
  BookTicker, AggTrade, Depth, OnlineNormalizer y los siete unit tests. Los
  bloques decisivos son 174–218 y 221–285; I1 timestamp AggTrade permanece abierto.
- `crates/data-pipeline/src/ws_client.rs`: 1–210 y 255–469; el resto se recuperó
  pero no se utiliza como inspección semántica completa. Blob
  `7e9ec19d593828e9eb76bb12b5c5dc4335f50c35`.
- `crates/data-pipeline/src/lib.rs`: 1–29; reexport público.
- `crates/data-pipeline/tests/parser_input_contract.rs`: 112–148, contratos Depth.
  Los controles de keys espaciados fuerzan fallback; no cubren la frontera
  canónica más espacio interno ni b/a.
- `src/parsers.rs`: leído completo; alcance evaluado 1–113 y tests Depth
  desde 267. Blob `1f062638da3455d3f423d46c3165979b2f18a678`.
- `src/bin/god_engine.rs`: 863–893, 2990–3080, 3230–3260; línea 3201 por búsqueda
  directa. Blob `eea831560cce74f00570b230724f5eef3f106420`. No revisión integral del bin.
- `crates/quantum-arena/src/state.rs`: 748–767; `core/lib.rs`: 1537–1579;
  `calibration.rs`: 324–345, cierre de función conocido por informe root.
- Runner 1–94, wrapper 1–33, recibo correcto completo, logs correctos completos
  y log de compilación fallida anterior. No se ejecutó ningún binario de evidencia.
- Binance: sección Partial Book Depth Streams, líneas extraídas 624–817;
  REST Order Book, 3049–3205. Consultado 2026-10-08. No se descargaron feeds.

Blobs adicionales del mismo commit para reproducir esas lecturas:

| Ruta | Blob Git |
|---|---|
| `crates/data-pipeline/src/lib.rs` | `e5de1bba8b6512f81510a776725d81406c160359` |
| `crates/data-pipeline/tests/parser_input_contract.rs` | `a9aa6231e8d778b4c4d9b6652042126d2c0bd343` |
| `crates/quantum-arena/src/state.rs` | `a9b0813d22d967458416c2e5b8c050cdb014a211` |
| `crates/god-engine-core/src/lib.rs` | `b045efbf82766090478c9da17412b05acb7d2810` |
| `crates/god-engine-core/src/calibration.rs` | `3fb9599aa81ac414e56f0cb68b1131f5258a630e` |

## Ajuste documental sugerido antes de publicar

Reemplazar el primer párrafo de «Productor, consumidores y límites» del borrador
`REVISION_PARSER_DEPTH_R4_2026-10-08.md` por una distinción explícita:

> La implementación de BinanceStreamer enlaza depth10 con DepthEvent y éste
> puede publicar sumas al arena; sin embargo el esquema USD-M oficial usa b/a,
> mientras DepthEvent exige bids/asks. Además, no se encontró instanciación
> productiva de BinanceStreamer en los Rust versionados del candidato. El bin
> god_engine tiene un parser separado que sí consume b/a. R4-I6 reproduce la
> mezcla de lados para la API bids/asks; esta revisión no demuestra que ocurra
> en el feed activo. La incompatibilidad de esquema requiere un contrato
> productor–consumidor propio antes de activar o modificar esa ruta.

Mantener la referencia al consumidor del veto como impacto **condicional**,
sin eliminar ni suavizar vetos con esta evidencia. El siguiente lote debe
reservar la ruta, fijar el esquema y separar prueba de parse, publicación al
arena y consecuencias por composición. Todo permanece abierto.

## Comprobación adicional del paquete documental

Paquete leído: `publication_packet_2026-10-08`, manifest de 25 archivos, todos
los hashes coincidentes en esta revisión. Comparación contra segmentos Sol
de `7acf36aa`: §11 del plan por archivo (11904 bytes), adenda sincronización
(2334 bytes) y nota del plan Quant (1858 bytes), los tres **byteidénticos**.
Hashes respectivos: `bfbebd35d4d1efa32520a397b7af95d1c34e0e50b4b2fc47d8d018553054eeac`,
`15ebc43fbfb21cf165534e44a69aebf1ba01ec46450d08c9c433a0b361f4409a`,
`447a7a9c9fb14589a50df903d45e975520612cf1977504ad6e7c63f01c35dba3`.
No quedan placeholders CODE en esos Markdown; T1_STATE y VALIDATION_STATE
siguen pendientes como corresponde a un paquete todavía sin publicación.
