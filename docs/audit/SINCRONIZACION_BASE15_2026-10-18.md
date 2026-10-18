# BASE15 — sincronización de plan e inventario con evidencia privada delimitada

Corte documental: 18 de octubre de2026, 09:48 America/Bogota. Referencia main `0db937040b3616980bfdb500cdcda4e2d01d080a`; rama de esta publicación `codex/base15-doc-sync-20261018`. ROOT solicitó esta actualización de los documentos propios mientras ejecutaba RiskRED. **Esta publicación no modifica Rust ni incorpora el CONTROL privado por ancestría.** El maestro compartido de AGY y las filas/apéndices de otros agentes se conservan.

## Plan e inventario conservados

Main0db no contenía los tres originales BASE15 de la rama `cff8886b5c50ff99e377b30082c2c3c814c91cf1`. Se publican el [plan BASE15](PLAN_BASE15_2026-10-15.md), el [inventario](INVENTARIO_BASE15_2026-10-15.json), la [cola](COLA_BASE15_2026-10-15.tsv) y sus dependencias documentales: [Git histórico](GIT_BASE15_2026-10-15.json) y [validación histórica](VALIDACION_BASE15_2026-10-15.json). El plan recibe un aviso de alcance, el protocolo actual y una adenda; los otros cuatro se conservan byteidénticos a sus blobs de cff8886.

El inventario y las1.863 filas siguen ligados al snapshot `b9cd773538fae9467751883eafdf3ee6ed5142f9`:13 rutas con hallazgo abierto,1 en revisión y1.849 sólo inventariadas; **cero archivos completos certificados**. No se regeneró el censo para main0db ni se añadieron resultados de un blob privado a una fila que identifica otro blob. Las categorías/propietarios propuestos y las evidencias históricas se conservan. El registro JSON suplementario de esta ola no equivale a cobertura del árbol actual.

## Core07: un par privado real, sin herencia a main

La [auditoría Core07 ya publicada por ROOT](CORE07_CEROS_REPLAY_RED_GREEN_2026-10-18.md) describe RED3PASS/2FAIL → GREEN5PASS/0FAIL, mismo corpus de cinco contratos,12 invariantes y8 guardas verdaderas. RED pertenece al commit privado `e4705ec93366e616f51d6f1ceaecde52b3ce7a58`, tree `8e835a5878fb48a148fc8ef3c2d94820b4c24e31`; GREEN al privado `084dcaf4fa88733bf62705dd8c2411215f039cea`, tree `25561b01ede699f7f3b29bbddd6fdcb643c19dd9`.

Corpus SHA256 `17e599ed1b37d8051a8644ef698dcd45a28710c634a7051240c8b362e41c6d54`. Archivo durable independiente12.18MiB, manifest `15410d8e06efb2d750337769e9a7da5175279a812de8bd76f6e5e8bfc9692b55`; la auditoría enlazada enumera originales, binarios retenidos, logs, recibos y límites. Esta ola contrastó el documento inmutable de main0db, no volvió a certificar los originales binarios.

RED observó fallos negativos en Dual y Depth; el caso negativo Trade no se alcanzó. GREEN completó el bucle Depth→Trade. Cargo RED101 se distingue del wrapper86; GREEN terminó0/0. Los coordinadores liberaron después de limpieza explícita de auxiliares propios VCTIP, no por drenaje espontáneo. RED tuvo solapamiento con un Cargo ajeno posterior a la adquisición: las guardas válidas no acreditan aislamiento global. El recibo original de all-targets0/86 sigue inválido; el check posterior válido tiene su propia fuente y no se reescribe.

Esto cierra **1/7 pares privados**, no el archivo Core completo, toda admisión temporal, LG5 ni el runtime público. El puerto autónomo de dos guardas y el corpus público requieren aplicación y pruebas en la composición final preservando los cambios nuevos de main.

## Cola causal al último aviso recibido de ROOT

| Frente | Estado documentado | Límite |
| --- | --- | --- |
| Core07 | Par privado real cerrado3/2→5/0 | Publicación runtime y validación del puerto público pendientes. |
| Risk01 | ROOT comunicó RED13 real, coordinator61604/Start14:30:20.1551800Z, Cargo57388/Start14:30:36Z, token70dab8ca-5ee3-48a5-b88a-5680977f1bf5; sin resultado recibido |13 es el corpus anunciado por ROOT, no13 fallos/aprobaciones inferidos. Este documento no inspecciona procesos ni concede lease. |
| Kelly02, Flow03, Hidden04, Replay05, Reporter06 | Otros cinco pares sin resultados causales recibidos | Ningún GREEN ni cierre por proximidad con Core07. |
| E2E integral, HOST/R4, T1 de la composición ROOT | GREEN integral no observado; gates propios pendientes | T1 de Fable u otras fuentes no se hereda por el compuesto. |

ROOT conserva la ejecución y el binding por fuente/corpus. La excepción de recursos actual del lanzador ROOT admite2GiB físicos libres/jobs1 por la orden del operador de trabajar con RAM disponible; no modifica runner236, parámetros ni12+8 y no autoriza compilación paralela desde BASE. Se preservan target y lock canónicos. No se fija una guarda4GiB que contradiga esa instrucción ni se pide cerrar aplicaciones.

## Dependencias que la composición debe resolver

**Flow03/B15-05 y recursión F-A6.** Se incorpora la [adenda específica](ADENDA_FLOW03_FA6_BASE15_2026-10-18.md), derivada de la propuesta externa SHA256 `9b8ed106ab5dd6d98848639154d54790108e3d5271c023aa785b9074eb89ff4b`. La consulta recurrente permite conceptualmente transportar la normalización O(1) y la referencia finita; no permite portar íntegro el oráculo que recalcula la historia con alpha nuevo y corte5/beta. Conservar n_seen, origen0, igualdad/rechazo regresivo, segundos y la frontera de beta nueva desde last_ts. Renovar fuente y preparar corpus público compuesto separado; el corpus privado sellado permanece intacto. La nota original observó WIP histórico; esta publicación no lo convierte en freeze ni PASS de composición.

**Risk01 y CL135 público.** En main0db, `crates/risk-engine/src/lib.rs` expone `MedidaOrden` con campos privados, `tomar_medida_validada(&mut self, coin_id)->Option<MedidaOrden>` y la función asociada `RiskEngine::registrar_medida(arena,&medida)`, sin self. El contrato es tomar al abrir, conservar con la posición y registrar al cierre elegible; Core usa `MedidaDeEntrada` con generación/lado. `registrar_orden_abierta` ya no es la API pública actual. El puerto de Risk01 debe preservar y probar ese protocolo, no sustituir nombres o construir campos privados. La prueba privada13 no acredita por sí sola el adaptador público. Blob Risk de main0db: `d912600cda69c6d5a0972ec703b76f637de14853`.

**LG5.** HIGH01/02 continúan abiertos bajo condiciones de ausencia de soporte, cierre pendiente elegible y aporte posterior; restauración robusta y veto inmediato intencional no resuelven adquisición por cohorte. MED06 conserva la obligación de identificar/justificar transferencia entre políticas; generation de posición no es generation de genoma. MED04 es un límite inferencial documentado: N1/sd poblacional0 no constituyen una LCB. No se propone resetear contadores, fabricar N/pagos o suprimir el veto EV. Expediente RO BASE15: manifest `8a653e1962a2e8208bc2f1a0563fb6242fde3e9f0a39b02f30df702c79b28f32`.

## Próxima aceptación y límites de esta ola

1. Esperar y clasificar el terminal Risk01 real, sin atribuir resultado por su lanzamiento o compilación de Polars.
2. ROOT continúa cada par con su fuente/binding/corpus y turno; renovar los puertos sobre main actual y preservar CL135/F-A6 y demás aportes.
3. Comprobar la composición pública por su propia evidencia antes de integrar runtime. Mantener PR65/BASE, Quant, SI, Hawkes y Fable reservados hasta su relevo explícito; no borrar ramas/worktrees activos por esta publicación documental.
4. Seguir el censo archivo por archivo con pruebas ligadas al blob que realmente se revisó. Los resultados parciales privados no son certificado integral ni evidencia de +100% cada72h.

En esta ola se hicieron comprobaciones documentales y de integridad del inventario. Cero Cargo/rustc/pruebas Rust/modelos/CIM/acciones sobre procesos; ningún acceso al índice ROOT. Los Git de esta publicación operan sólo en el worktree documental y sobre objetos inmutables; el merge pendiente de BASE se conserva. La publicación y sus hashes finales se registran externamente para evitar autorreferencia del commit.
