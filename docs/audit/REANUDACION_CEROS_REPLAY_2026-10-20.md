# Reanudación de ceros y replay — 20 de octubre de 2026

ROOT retoma las correcciones sobre el main vigente. El CONTROL privado y sus recibos se conservan como evidencia; no se fusiona esa rama ni se transfieren sus resultados al candidato público. Este documento fija el estado que se conoce, el reparto con los otros agentes y la secuencia de validación pendiente. No certifica archivos completos ni rentabilidad.

## Estado conservado e interrupción

Los cuatro pares privados históricos siguen cerrados: Core07 RED 3/2→GREEN 5/0, Risk01 6/7→13/0, Kelly02 0/2→2/0 y Flow03 1/6→7/0. Cada ejecución conserva su corpus, recibos y límites causales; no son una suite del main actual. El detalle previo está en [el cierre de cuatro pares](CIERRE_CUATRO_PARES_PRIVADOS_2026-10-18.md).

Hidden04 conserva el GREEN 16 numérico con driver/validator 0/0 y 12 invariantes + 8 guardas verdaderas. Recibo SHA256 `f2d55fffac19f713b243c25af8a6c94f1d1a27f81592ed2e241c2d1b4ebd9e3b`; log `e9aeec9b5ed33bc105dc6d5ed8cd0f7830865eca92e3f7178eec40a2f3ae6485`. No existe su COORDINATOR_RECEIPT final. El owner 69896, hijo 67480 y auxiliar VCTIP 5860 ya no aparecen; la sesión 61831 desapareció y el lock actual está ausente. Estos hechos no acreditan cierre natural, drenaje observado ni una liberación recuperada. El boot observado corresponde al 13 Oct; no se atribuye un reinicio a la interrupción.

El GREEN 2 de publicación en Core nunca corrió. Hidden continúa incompleto y no incrementa el total histórico a 5/7. Replay05 y Reporter06 privados pendientes también quedan INCOMPLETE; no se maquillan como listos. Se conserva el check privado previo como antecedente de su fuente, sin heredarlo a main, a CI o a otro grafo de dependencias.

La rama privada permanece en `e2a2fb72bad63de1aeb06670dfaecbb5ee2f0b1f`, árbol `6ec292db524273b9d2b9a4a6e4ea45e5564d3151`, limpia según el snapshot fresco comunicado por ROOT. HEAD, árbol, rama e índice lógico `32f395883fcf2a333f8b329037f590bb69ccc025cd43a72eafd7344ba6c2068d` coinciden con los anteriores. El índice raw actual es `50e58bd4d5096504d0ac3c4987d27fae22f9b23e078fb84e1feb1b8ffca57e47`; el histórico era `da951cef560871753f2f762f232f3c09417c6d14a17e595823c853fa50b9d1e2`. Se mantienen separados: no se atribuye causa ni se afirma un cambio sólo de metadata sin comparar la copia antigua. El archivo con el pin viejo abortó antes de crear carpeta y sin modificar Git o fuente; no se reparan recibos. Ya se contrastaron los 771 hashes de inputs compilados contra el GREEN original y los 17 fingerprints del corpus. El [archivo privado incompleto](C:/Users/jhona/.codex/audit-evidence/trader-gemini/hidden04-private-incomplete-20261020-9fe46a9a-fe7c-4cbe-af50-747dd18dadaf/README.md) conserva 139 archivos (9878187 bytes; aproximadamente 9,878 MB), manifest SHA256 `333b71a6daa473d20ff477b37ea16a8738b45aa02920168ffe365b58a45ae9a0` y verificación `2143ff6fd61b3881bb46c468d2d34320dafedd4fbfac8bc8d9327fbaaefb47b1`. Fuente e índice raw actual 50e58 permanecieron iguales antes/después DURANTE el archivo, sin comparar ni declarar iguales los bytes raw históricos da951 durante la pausa. No hubo manipulación de lock, fuente o refs ni Cargo; no se inventó un recibo de coordinador ni un resultado nuevo. La publicación documental deberá revalidar el snapshot actual; este texto no sustituye una lectura nueva.

## Base pública y coordinación

La propuesta documental parte de `597de4af0efae7c64c34b618651c357d406d38c3`, servidor y tracking observados. Main local `d7094c6788171170fba8e9eb3b165e28e92be87f` está cinco commits detrás y ocupado; no se ha ejecutado FF en este corte. El censo reciente registra 12 ramas locales, 6 remotas reales y 7 worktrees. No hay ramas borrables: Fable integrado sigue ocupado y las restantes no-main conservan exclusivos. Antes de publicar, ROOT comprobará otra vez el main real y reanclará si avanzó.

67dae483 revirtió las filas ROOT y BASE recién publicadas en 5f11. Latest conserva esas filas viejas. Esta entrega actualiza sólo la fila ROOT y añade información al final de MEMORIA y del buzón. No restaura filas antiguas de BASE, AGY, Sol ni otros agentes, y no reescribe el censo histórico.

BASE mantiene el plan maestro y el inventario canónico histórico de 1863 rutas. El recenso actual de 2053 rutas y delta 424 pendiente está preparado sobre 597de4af, aún no publicado; este documento no afirma una actualización ya integrada del canon. ROOT aporta cierres concretos a ese registro; no abre otro inventario que compita ni marca un archivo completo como auditado por tener un test. SI conserva el carril EV de sólo lectura fuera de las siete rutas. F7 toma la entrega independiente de scripts y la primera CI cloud autorizada. ROOT no compite por Cargo ni CI mientras ese carril trabaja.

La observación de recursos a 11:27Z fue 0,415 GiB físicos libres y 2,636 GiB de margen de commit, con cero Cargo/rustc/VCTIP. Es una medición puntual. La instrucción del usuario es trabajar con lo disponible; no se pide cerrar aplicaciones ni se repite una aprobación ya dada. Se avanza en preparación, revisión y ejecución coordinada con evidencia de la fuente real.

## Puertos autónomos sobre main actual

En dos días cambiaron Core, Risk, Flow, replay y el grafo Cargo. Revalidar CONTROL antiguo no es el siguiente hito: los siete cambios se reanclan de forma acotada sobre una rama pública propia desde latest main. Se preservan CL135, F-A6 recurrente, F1-101/F1-061 del replay, PR86 del Core, NoReplay y las demás modificaciones integradas. Ninguno de los puertos de esta tabla tiene una nueva ejecución Rust en este corte.

| Puerto | Decisión y obligación actual |
|---|---|
| Core07 | Portar las dos guardas temporales por activo y renovar los contratos sobre el Core actual. Preservar los cambios de constructor/modelos, cierre y PR86; no sustituir el archivo completo. |
| Risk01 | Introducir reloj causal del evento sólo para el cálculo de carry y adaptar su caller, conservando el punto de consulta legacy cuando no se proporciona reloj, origen cero, CL135 y validación actual de ATR/precio. Nueva evidencia pública propia. |
| Kelly02 | Reespecificar el contrato sobre F1-116: LH04 default ahora empieza con mínimo 1, no 10. N9 ya no justifica excluir esa ruta. No heredar el resultado de un fixture anterior ni cambiar su premisa en silencio. |
| Flow03 | Mantener los mismos cuerpos de 8 contratos recurrentes con presión real Hidden + 2 de alcance propio de ratio/referencia y hospedarlos en integración de Core. Core ya depende de Signal, DarkAlpha, Omni y Strategy: no reintroducir DarkAlpha en Signal ni agregar una arista sólo por el hospedaje. Esto prueba contratos de composición, no Core E2E ni teoría matemática nueva. |
| Hidden04 | Separar los tres mundos A0/B1/B2 descritos abajo, con corpus público idéntico entre ejecuciones. Conservar los originales privados. |
| Replay05 | Renovar span 8 y ledger 5 preservando ambos cierres por tick de F1-101, la autoridad de sizing de F1-061, trade-only y marcas terminales actuales. Misma preimagen para ambos RED antes del cambio de runtime. |
| Reporter06 | Preservar los 21 contratos públicos NoReplay y añadir los 2 contratos de reporte; comprobar reporte contra el replay real y conservar el union 23. No reemplazar el archivo público por la versión privada. |

## Hidden: tres mundos causales

La separación impide atribuir a un arreglo lo que sólo demuestra el siguiente. Los nombres indican una secuencia propuesta; todavía no son resultados públicos.

1. **A0:** main con runtime Hidden `cc526…`, añadiendo únicamente los contratos numéricos 16, publicación 2 y tiny-z U1. Exigir RED semántico del lote 16 y publicación 2, con controles positivos alcanzados. Los tests U1 forman parte del mismo corpus, pero su atribución específica a tiny-z requiere el RED en B1; un fallo previo de control en A0 no demuestra ese defecto adicional.
2. **B1:** aplicar sólo fix04 para runtime `6fed…`. Exigir GREEN 16 y GREEN 2 de publicación, y RED auténtico de los 2 contratos U1. El GREEN 16 privado previo no sustituye estas ejecuciones.
3. **B2:** aplicar únicamente el arreglo U1 adicional, hacia la imagen final `6b5eef58…`. Exigir GREEN 2 U1 y la suite Dark completa, además de las regresiones del candidato final.

El antiguo arnés público de publicación Core `bb48…` inicializa `scalp_forest: None`, campo retirado del Core actual. Se adaptará exclusivamente eliminando esa asignación obsoleta, con nueva huella de corpus constante en los tres mundos. No se cambia el cuerpo de las aserciones ni el original privado. Una modificación del arnés o error de compilación no cuenta como RED del sistema.

## Validación, integración y cierre

La preparación y la CI inicial de F7 se mantienen como entregables independientes. ROOT revisará sus scripts y resultados dentro de su alcance antes de validar el candidato público. Cada ejecución debe identificar commit, árbol, corpus, comando, resultado y límites; no se heredan PASS de otro agente o de otro árbol. CI cloud no se describirá como prueba de las 12+8 guardas Windows locales.

El candidato público final requiere check de workspace con todos los targets, contratos y regresiones propias de los siete puertos y backtest completo `--lib --tests` con al menos 110 PASS reales agregados y cero FAIL. T1 genético declara dos tests; la cobertura de genes y el número de pruebas son medidas distintas. No se contarán genes, líneas de log, tests filtrados o ejecuciones de otro árbol como 110 PASS.

Tras el resultado válido se revisará el diff contra latest main, se resolverán conflictos semánticos, se integrará sólo lo validado y se comprobará el commit y árbol en el servidor. Si main avanza, se conserva el trabajo y se valida la composición resultante. Las ramas propias se retirarán tras acreditar integración y liberación; ramas ocupadas o con exclusivos se conservan. El cierre se comunica a BASE, F7 y SI y se refleja en el plan maestro.

La revisión total del sistema, LG5 HIGH01/02/MED06 y la meta de duplicación neta compuesta cada 72h siguen pendientes. Los contratos de software no prueban rentabilidad, consistencia económica ni viabilidad de 100% cada tres días.
