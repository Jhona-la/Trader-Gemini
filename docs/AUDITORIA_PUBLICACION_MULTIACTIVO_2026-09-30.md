# MP — Publicación multiactivo, identidad y coherencia del grafo de modelos

## 1. Dictamen y límites del corte

Fecha: 2026-09-30. Base: `ee438edb03333b67e2e41d8ba31229e8350a505e`.
Rama: `codex/model-publication-contract`, checkout Codex aislado. MR permanece
en su rama y [PR23](https://github.com/Jhona-la/Trader-Gemini/pull/23), todavía
pendiente de CI/revisión al inicio de esta pasada. No se mezcla su publicación
con la de MP ni se atribuyen sus cinco tests nuevos al conteo de esta rama.

**Dos reparaciones candidatas locales, cinco expedientes abiertos.** Los
expedientes no son siete bugs reparados ni siete descubrimientos independientes:
MP-07 remite expresamente a MR-03. Los cinco fallos de tests corresponden a dos
familias causales. No se certifica todo el proyecto, el motor vivo ni su retorno.

El corte base contiene 1.377 archivos versionados y 434 Rust. Es inventario,
no cobertura semántica. Se inspeccionó la ruta de carga/publicación/consumo y
sus contratos, no cada uno de esos archivos. No se ejecutó el motor, training,
promoción, evaluación T-1 ni tests de exchange; sólo modelos sintéticos temporales.

La teoría pertinente aquí es la composición de actualizaciones, coherencia de
snapshots y procedencia del artefacto. Añadir una ecuación de física o un nombre
cuántico no prueba ninguna de esas propiedades. La utilidad de una futura
teoría debe expresarse como hipótesis, unidades, observable, contrato y contraste
fuera de muestra, no como complejidad nominal o promesa de +100 % cada 72 h.

## 2. Grafo causal: desde el archivo hasta el veto

```mermaid
flowchart LR
  J[JSON fuente] --> P[Resolver pareja de rutas]
  B[BIN cache o legacy] --> P
  P --> V[Parseo y validacion estructural]
  V --> U[Publicacion por clave]
  U --> M[Mapa compartido de modelos]
  M --> C[Snapshot del modelo por activo]
  C --> R[has_roster_model y prediccion]
  R --> G[Puertas de entrada existentes]
  W[Watcher por timestamp] --> P
  E[Version y evidencia posterior] -. contrato aun incompleto .-> U
```

No se ha cambiado la política del veto de roster. Se repara una causa técnica
por la que un modelo que acababa de publicarse podía desaparecer del mapa.
Sin modelo, el consumidor cambia a su lógica de ausencia/arranque frío; no se
debe atribuir automáticamente ese resultado al genoma, a la señal o al régimen.
El posible efecto sobre decisiones es una ruta de código comprobada, no una
cuantificación de pérdidas o de frecuencia de la carrera en producción.

## 3. Matriz de expedientes

| ID | Prioridad | Estado en este corte | Contrato afectado |
|---|---|---|---|
| MP-01 | P1 | candidato reparado; RED reproducido | dos escritores pueden perder el activo del otro |
| MP-02 | P1 | candidato reparado; RED reproducido | pareja de rutas y preservación de la fuente |
| MP-03 | P1 | abierto, análisis estático | orden de publicación no es orden de generación |
| MP-04 | P1 | abierto, análisis estático | múltiples cabezas no forman una versión transaccional |
| MP-05 | P2 | abierto, análisis estático | escritores directos pueden saltarse el contrato |
| MP-06 | P2 | abierto, traza lógica de fuente | timestamps JSON/BIN causan recargas repetidas |
| MP-07 | P1 | abierto, referencia a MR-03 | mtime y estructura no vinculan JSON al predictor servido |

## 4. MP-01 — Pérdida de publicaciones entre activos

**Evidencia.** En la base, `NanoForest::load_global` y `store_global` leen
`GLOBAL_FORESTS`, clonan el mapa, insertan una clave y hacen `store` del mapa
completo. El reemplazo del puntero es atómico; la secuencia leer-modificar-
escribir no lo es. Es un fallo de actualización lógica, no una violación de
seguridad de memoria. Afecta a las dos APIs, con independencia del símbolo.

**Intercalado mínimo.** Sean M el mapa original y A/B activos distintos.
El escritor A calcula M_A = M[A := modelo_A]. Antes de publicarlo, B calcula
M_B = M[B := modelo_B] a partir de la misma versión. Si se publica M_A y
después M_B, el resultado no contiene la actualización A. Cada lector ve un
mapa completo y válido; eso no evita perder trabajo de otro escritor.

**Invariante requerido.** Para claves distintas, las actualizaciones deben
componer: U_A(U_B(M)) = U_B(U_A(M)). El resultado debe contener ambas, además
de las entradas anteriores que ninguna operación pidió retirar. Ésta es una
condición de concurrencia multiactivo, no una condición financiera.

**Reproducción ejecutada.** `mp_concurrent_store_preserves_every_distinct_asset`
lanza 32 escritores sincronizados por barrera durante ocho rondas, con una
clave única por escritor/ronda. En la ejecución RED se perdieron 183 de 256
publicaciones. `mp_concurrent_load_preserves_every_distinct_asset` prepara 24
pares JSON/BIN independientes fuera de la contención: se perdieron cuatro
altas pese a que las 24 llamadas retornaron éxito. Los números dependen del
planificador; no son tasas esperadas ni un benchmark. Las aserciones requieren
cero claves ausentes y el valor exacto de cada modelo.

**Reparación.** Se centraliza `load_global` en `store_global`; éste utiliza
`ArcSwap::rcu`. Si otra publicación cambió el mapa entre lectura e intercambio,
el cálculo se repite sobre la versión actual. El parseo, la validación y la
construcción del `Arc<NanoForest>` quedan fuera del cierre reintentable. El
cierre sólo clona el mapa y la clave e inserta una referencia al modelo ya
validado; no repite E/S, entrenamiento, telemetría ni efectos externos.

**Fundamento verificado.** Se leyó la implementación local de la versión 1.9.2
fijada en `Cargo.lock` y su [contrato oficial de RCU](https://docs.rs/arc-swap/1.9.2/arc_swap/struct.ArcSwapAny.html#method.rcu).
La documentación advierte del intercalado load/store y de reejecutar el cierre.
Firecrawl se usó para contrastar esa fuente primaria, no para atribuir una
garantía financiera ni una latencia no medida al cambio.

**Preservación.** El lector que conserva un `Arc` anterior sigue viendo el
modelo anterior; lectores posteriores pueden ver el nuevo. Una carga inválida
no reemplaza el último válido ni crea su caché. Esos dos contratos ya pasaban
en la base y se conservan: no se cuentan como nuevos arreglos.

**Límite.** La garantía cubre escritores que usan estas APIs. No da exclusión
mutua sobre el archivo, orden de generación para la misma clave, promoción
estadística ni snapshot conjunto de todas las cabezas del activo.

## 5. MP-02 — Rutas de caché que alteran padres o destruyen la fuente

**Causa.** `str::replace` sustituía TODAS las apariciones de `.json`/`.bin` en
la ruta, no sólo la extensión final. En `archive.json/asset.json.v2.json`
se transformaban directorio y stem. En la dirección inversa, la ruta de un
BIN bajo `archive.bin` podía buscar un JSON en otro directorio, concluir que
no había fuente más reciente y aceptar el caché obsoleto.

**Reproducciones.** El test de directorio `.json` no encontraba la caché
hermana que debía crearse. El test de directorio `.bin` preparó explícitamente
mtime(BIN)=epoch+10.000 s y mtime(JSON)=epoch+20.000 s, sin sleeps: el loader
devolvió bias +2 del BIN cuando la fuente hermana nueva tenía bias -2.
Estos valores son fixtures de identidad, no probabilidades calibradas.

**Destrucción adicional.** Para una fuente llamada `model`, el reemplazo de
`.json` era un no-op: fuente y destino de caché eran el MISMO archivo.
La carga retornaba éxito pero convertía el JSON original en bytes bincode.
El contrato RED comparó bytes antes/después y demostró esa sobrescritura.
El experimento sólo afectó a un directorio temporal propio, eliminado por
RAII incluso al fallar la aserción; no se modificaron modelos del proyecto.

**Reparación.** Para extensiones reconocidas se utiliza `Path::with_extension`,
que conserva padres y stem. Los nombres no convencionales continúan admitidos
como fuentes JSON, con la misma validación, pero sin caché implícita. No se
inventa un destino que pueda colisionar con la fuente. Se mantiene el soporte
legacy de BIN sin JSON y el fallback de BIN corrupto a JSON válido.

**Pruebas adicionales.** Nombres `.weights`, `.JSON` y un directorio `.json`
con archivo `.payload` permanecen byte a byte intactos. Un BIN con parámetro
infinito sigue rechazándose. Ninguno de estos casos autoriza usar un modelo
sin evidencia de entrenamiento/promoción.

**Límite.** Los hardlinks/symlinks pueden hacer que rutas textualmente distintas
apunten al mismo archivo; no se implementó una política antialias de filesystem.
Tampoco se convirtió la escritura de caché en transacción ni se reemplazó el
criterio mtime por identidad criptográfica. No afirmar que MP-02 cierra MR-03.

## 6. MP-03 — Un escritor atrasado puede volver a publicar una versión vieja

**Estado:** abierto; diseño/API observado, no incidente productivo atribuido.
`NanoForestData` no contiene generación, identidad del dataset, símbolo,
aprobación ni predecesor esperado. `store_global(key, forest)` acepta cualquier
bosque estructuralmente válido y no compara una generación monotónica.

Un escritor puede terminar de preparar V1, quedar suspendido, mientras otro
publica V2, y luego publicar V1. RCU preserva los demás activos, pero para esa
clave prevalece el último intercambio, no el modelo más reciente o aprobado.
Un SHA distinto tampoco determina cuál versión debe dominar.

**Cierre requerido.** Contrato de promoción con versión/linaje por clave,
compare-and-publish contra generación esperada y política explícita de rollback.
La decisión no debe inferirse de mtime, orden lexicográfico del hash o calidad
medida sobre otra muestra. Probar llegada invertida y rollback autorizado.
No se añadieron versiones ficticias ni se alteró la promoción ajena.

## 7. MP-04 — Snapshot individual no equivale a ensamble coherente

**Evidencia.** `GodEngineCore::process_event` obtiene `_MOTOR`, `_VOL` y `_OI`
con llamadas independientes a `get_global`. La predicción direccional y su
base sí proceden del mismo `active_forest`: no se inventa una mezcla allí.
Sin embargo, otras cabezas pueden capturarse después de otra publicación.
Además, el productor carga cada archivo por separado, no una generación de
ensamble. Capturar un solo mapa evita una mezcla entre lecturas, pero no evita
que el propio mapa contenga una transición parcial entre varias cabezas.

**Impacto posible.** Una decisión combina outputs de distinta procedencia sin
receipt que indique la combinación. No es intrínsecamente inválido si el diseño
permite cabezas asincrónicas; falta declarar y verificar esa compatibilidad.

**Cierre requerido.** Elegir entre versiones independientes compatibles o bundle
transaccional. Para lo segundo: validar bundle completo, publicarlo una vez,
capturar un snapshot por decisión y registrar IDs. Para lo primero: publicar
evidencia, edades y esquemas por cabeza. No desactivar predictores por ausencia
de una sincronía universal arbitraria ni afirmar sincronía perfecta por usar Arc.

## 8. MP-05 — La API pública permite saltarse la publicación segura

`GLOBAL_FORESTS` sigue siendo `pub static ref`. En el árbol leído hay escrituras
directas en helpers de tests de `ml_inference.rs` y `god-engine-core/src/lib.rs`.
No se encontró otro escritor productivo directo en la búsqueda `crates/ src/`;
esa búsqueda no demuestra ausencia en consumidores externos al workspace.

Una escritura incondicional ajena a RCU aún puede reemplazar un mapa antiguo.
La encapsulación privada del registro y una API explícita de fixtures reducirían
esta posibilidad, pero son un cambio de API que no se impone en esta reparación.
No se eliminan modelos globales desde los nuevos tests: usan namespaces propios
en su ejecutable aislado y no interfieren con activos reales.

**Cierre requerido.** Auditar consumidores, retirar acceso mutable directo de
forma compatible y probar la frontera pública. Las pruebas secuenciales no
acreditan aislamiento de toda la suite con múltiples escritores de fixtures.

## 9. MP-06 — Oscilación de fechas del watcher sin cambios de archivos

**Evidencia estática.** En `src/bin/god_engine.rs`, la carga inicial prefiere
JSON y usa BIN sólo si no existe JSON; el loop posterior recorre AMBOS. La
tabla `forest_timestamps` está indexada por file stem, no por ruta ni contenido.
Con `t_json != t_bin`, una pasada actualiza ese mismo slot a `t_json`, después
a `t_bin`. En la próxima pasada ambas comparaciones vuelven a diferir.

**Traza lógica:** estado inicial t_bin → visita JSON → t_json → visita BIN →
t_bin → siguiente ciclo repite. Es válida con orden estable de enumeración,
sin modificar los archivos. No se ejecutó el host para medir su frecuencia.

**Impacto.** Relecturas, deserialización, validación y clonación de mapas pueden
repetirse en cada ciclo del watcher (sleep de 10 s en el código leído). Además
de E/S/CPU innecesaria, genera mensajes de hot-swap sin nueva evidencia. Es
distinto del timestamp consumido antes de carga exitosa (MR-04); ambos surgen
de un estado de observación insuficientemente definido.

**Cierre requerido.** Un artefacto canónico por clave, estado de observación y
estado activado separados, actualizar el segundo tras éxito, reintentos medidos
y fixtures de dos pasadas sin cambios. Reservar coordinación con el dueño del
host antes de tocar su watcher. No imponer un nuevo cooldown literal para
ocultar el ciclo ni bajar frecuencia de auditoría como supuesto arreglo.

## 10. MP-07 — Identidad de caché sigue abierta (referencia MR-03)

El cambio de rutas conserva la política existente: BIN válido y suficientemente
reciente puede ganar sobre JSON. Es la discrepancia ya reproducida por MR-03,
no otro hallazgo nuevo. Igualdad temporal o caché futuro/copias restauradas no
vinculan el predictor al hash JSON del registro. Tampoco un hash autentica al
productor, acredita skill fuera de muestra o autoriza activar el artefacto.

El soporte de BIN standalone se preserva expresamente para no inventar una
política de migración. Cierre futuro: linaje de bytes fuente y payload, formato
versionado, validación de binding y receipt del predictor servido, manteniendo
una ruta de migración explícita. Un manifiesto diagnóstico no es ese receipt.

## 11. Coste, latencia y sentido de los límites

La ruta de predicción no recibe E/S ni un mutex nuevo. En escritura se conserva
la copia O(K) del mapa de K claves por intento, ahora repetible si hay contención;
los árboles se comparten por Arc, no se clonan por cada reintento. El tiempo
depende de contención, tamaño de claves, asignador y snapshots retenidos. No se
deduce una cota temporal dura ni nanosegundos por tick de esta estructura.

Debe medirse p50/p95/p99 de publicación separada de inferencia, volumen de
reintentos, memoria retenida y tasa de cambios reales. Los 32/24 escritores y
ocho rondas son parámetros de estrés de tests, NO límites del motor ni regímenes
de mercado. No se añadió cap de activos, timeout, umbral de rentabilidad ni veto.

## 12. Evidencia de verificación

Toolchain `nightly-2026-06-30`; `arc-swap` 1.9.2, lockfile sin modificar.
Ejecución Windows, perfil local unoptimized + debuginfo; CI usa debug=0.

| Ejecución | Resultado | Interpretación |
|---|---|---|
| Base, primeros ocho contratos | 3 pasan / 5 fallan, 0,56 s | RED real; build39,09 s |
| Candidato, diez contratos | 10 / 0 / 0, 0,29 s | GREEN; build34,84 s |
| Dos tests concurrentes ×20 | 40 ejecuciones aprobadas | sólo dos tests únicos, no sumar40 al total |
| Núcleo + cinco suites seleccionadas | 198 / 0 / 1 | build38,36 s; desglose abajo |
| Workspace all-targets/locked | éxito31,16 s | compilación, no ejecución de todos los tests |

Desglose disjunto: core lib146, conformal11, base1, ML12+1 ignorada,
MP10, atribución18 =198 aprobadas/0 fallidas/1 ignorada. La ignorada es
inventario manual de modelos guardados; no se habilitó. Las repeticiones
concurrentes y el primer GREEN no se suman de nuevo. La regresión de replay
se registra por separado al finalizar, sin inferir éxito por lanzar el comando.

Hash SHA256 de las fuentes probadas:

- `ml_inference.rs`: `72947A7685476162894366012DE24E6ACBF5F50E06B1EF8CBFA0861D907487D9`.
- `model_publication_contract.rs`: `42436619DB54F4683ABC20D961104AD0AA3A83BBB8723E42C6EA0F9B24CC6582`.

## 13. Integración, coordinación y orden de trabajo

MR/PR23 no se modifica por esta ola. PR10/20 siguen fuera de main remoto en
el corte observado; el merge local77d6632a de GLM no prueba push/integración
remota. No se borran esas ramas, MP, MR ni backups con commits exclusivos.
CX/PR22 sí está fusionada; su rama ya fue retirada con ancestralidad comprobada.

Aviso de reserva compartido en `.firecrawl/coordination-codex-causalidad-2026-09-29.md`.
No es acuse de recibo de Claude/GLM. Se solicita autorización específica para
publicar el nuevo payload MP en el repositorio público; no reutilizar como
si fuera ilimitada la autorización previa de MR. Merge condicionado a CI y
revisión cruzada del SHA final. Ninguna intervención en procesos de GLM.

Prioridad: validar/publicar estos dos contratos, cerrar promoción y linaje con
sus dueños, hacer explícitas las versiones de decisión y sólo entonces atribuir
resultados a teoría/genoma. Cualquier mejora de retorno necesita medición causal
fuera de muestra con costes; los tests presentes no contienen tal evidencia.

## 14. Cierre local verificado — código3f2be42d

La regresión complementaria finalizó con código0: biblioteca replay51,
paridad8 aprobadas+2 ignoradas, métricas21, labels25 y riesgo espectral3.
Total replay108/0/2; build2m03s, biblioteca77,35s y paridad35,79s.
Sumado de forma disjunta al núcleo/suites198/0/1: **306 aprobadas,0 fallidas,
3 ignoradas**. Dos ignoradas de paridad son experimentos manuales existentes;
la restante es inventario de modelos. No se habilitó ninguna ni se modificó
golden, fixture anterior o umbral para lograr aprobación.

Commit de código/pruebas/CI: `3f2be42d3e9effe102662cabb779405f70577fa9`.
Las adendas posteriores son sólo documentación. Se verificó conservación en
orden de1711 líneas de coordinación,8726 del maestro y1918 del atlas. Las
notas previas «replay en curso» son el corte inicial, no el estado final.

MR CI36718230501: check all-targets y contratos registry aprobados; replay
en curso al último seguimiento, no CI global aprobada todavía. Se publicó
[solicitud de revisión cruzada MR](https://github.com/Jhona-la/Trader-Gemini/pull/23#issuecomment-5912446341),
sin inferir respuesta. MP permanece local a la espera de autorización pública
específica; no se publica código o informe MP mediante ese comentario MR.
