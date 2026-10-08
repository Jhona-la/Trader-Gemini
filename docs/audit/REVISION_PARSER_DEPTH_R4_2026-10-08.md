# R4-I6 — Revisión completa del parser y mezcla de lados en Depth

Fecha: 2026-10-08. Revisor/ejecutor: Codex root. Fuente `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`,
blob `6c47744a94f5bea150e8a65c750b2a2f58c73cc4`; SHA256 `2eaf3e07f6ef1ead73d996c3fcff537354ca37b57b39a0ee3fe45e77cf07f5a7`.
Lectura completa del archivo `crates/data-pipeline/src/parser.rs`: líneas1–462,
incluidos sus siete unit tests; lectura completa del contrato externo de parser.
Estado **HALLAZGO_REPRODUCIDO_ABIERTO**, no cierre completo del archivo.

## Defecto y controles

`extract_wall_sum` termina en la primera secuencia literal `]]`, sin resolver
el cierre de su propio array. La selección fast exige sólo los dos keys
canónicos, no que todo el payload tenga formato minificado. JSON válido con
espacio entre corchetes de bids hace que su búsqueda llegue a asks; bids vacío
también recorre el array posterior. El lector suma cantidades del otro lado.

| Caso sintético, JSON validado por serde | Referencia bids/asks | Resultado del parser |
|---|---:|---:|
| Ambos arrays minificados, un nivel cada uno | 2 / 3 | 2 / 3, control correcto |
| Espacio entre cierres internos de bids; keys canónicos | 2 / 3 | **5 / 3** |
| Mismo espacio, key asks con espacio antes de `:` | 2 / 3 | 2 / 3, fallback correcto |
| bids vacío, asks con cantidad3 | 0 / 3 | **3 / 3** |
| asks primero, espacio entre sus cierres | 2 / 3 | **2 / 5** |
| Un nivel finito1e308 por lado, espacio en bids | 1e308 / 1e308 | **None**, overflow por mezcla de lados |

Son seis escenarios y seis aserciones de caracterización, no seis tests
del workspace. Tres comandos (versión, compile, ejecución) terminaron exit0.
La prueba compila el módulo completo sin cambios y enlaza rlibs cached de
memchr/fast_float/serde_json, con hashes registrados y `panic=abort`.
El último caso es de frontera numérica sintética; no representa tamaños de
orden observados. El [recibo](RECIBOS_DEPTH_R4_2026-10-08.json) incluye argv,
fuente, wrapper, dependencias y stdout. Un intento anterior no compiló por
incompatibilidad unwind/abort: no ejecutó el harness y no se cuenta como RED.

El fragmento del extractor es byteidéntico en adeb, f3 y e9, hash
`2e31b8acd3309b32d46ce402f96d7af9975031ce5625bd09456553c654a31119`. La reproducción ejecutada pertenece
a e9; la comparación estática de ancestros acota un defecto previo del
extractor, sin fingir ejecución de esos ancestros ni una regresión nueva
causada por el merge. No se modificó fuente, índice, HEAD, Cargo o T1.

## Productor, consumidores y límites

La implementación de `BinanceStreamer` enlaza depth10 con `DepthEvent` y
puede publicar sus sumas al arena. Sin embargo, el esquema USD-M oficial
usa `b`/`a`, mientras `DepthEvent` exige `bids`/`asks`. Además, la búsqueda
en todos los Rust versionados del candidato no encontró instanciación
productiva de `BinanceStreamer`. El bin `god_engine` usa un parser separado
que consume `b`/`a`. R4-I6 reproduce mezcla de lados para la API `bids`/`asks`;
no demuestra incidencia en el feed activo. La incompatibilidad de esquema
requiere un contrato propio antes de activar esa ruta. El
[informe independiente](DEPTH_INDEPENDENT_REVIEW_2026-10-08.md) detalla los
blobs/rangos y enlaza la documentación primaria REST y WebSocket de Binance.

La conexión condicional es `ws_client.rs:455–466` → `arena.update_l2_depth`,
`state.rs:754–770` → coins/tensor y `god-engine-core/lib.rs:1546–1566` → veto
del muro. `calibration.rs:327–347` compara muro contrario/favorable contra
el gen de razón. No acredita que el exchange emitiera el fixture ni pérdidas
en una operación real o replay completo. Comprobar esquema, activación,
slots, orden temporal y frecuencia antes de cuantificar incidencia. No se
ensayó red ni se modificaron vetos. Tampoco se afirma un build hermético:
el recibo identifica rlibs directos, sin inventariar todos los transitivos.

La compilación y las 19 pruebas del parser existentes pasan en 7acf; su fuente
es idéntica en e9. Cubren tokens, clocks BookTicker y mezcla de espaciado en
keys, pero no esta frontera estructural dentro de un array. Pasar esos
contratos no certifica el parser completo. I1 sigue abierto: AggTrade convierte
T presente inválido a0. Se inspeccionó también OnlineNormalizer, sin caller
productivo encontrado fuera de sus tests; su filtro de entradas no finitas
no es prueba de inmunidad a desbordamientos derivados de entradas finitas.
El scanner declara que no valida todo JSON/esquema; hay que fijar el dominio
aceptado y la política ante keys duplicados, sobre anidación y relojes.

## Corrección y siguiente lote

Reservar parser y consumidor con su dueño. Delimitar cada array propio con
profundidad estructural y tratamiento correcto de strings, o elegir fallback
antes de aceptar sumas provisionales. Keys canónicos no prueban formato
canónico del resto del mensaje. Mantener la validación finita por lado.

El contrato nuevo debe FALLAR antes del arreglo y pasar después, comparando
contra el parse estructural sobre permutaciones de espaciado/orden, arrays
vacíos, uno/múltiples niveles, wrappers `data`, límites numéricos y payloads
inválidos. Conservar controles nominales, BookTicker/AggTrade y overflow;
verificar publicación al arena y composición del veto en sombra. Medir
asignaciones/latencia: el camino serde asigna y «nanosegundos» no está medido
por este recibo. No integrar un fix de pipeline durante el T1 congelado.
