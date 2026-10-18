# Core07: par causal privado RED/GREEN real

Core07 completó un par funcional real: RED **3 PASS / 2 FAIL**, seguido de GREEN **5 PASS / 0 FAIL** sobre el mismo corpus de cinco contratos. Los recibos protegidos conservan **12 invariantes y 8 guardas de parámetros verdaderas en ambos mundos**. Esto cierra el par privado Core07 (1/7); no integra runtime en main, no valida todo LG5 y no certifica rentabilidad.

Archivo externo independiente: `C:/Users/jhona/.codex/audit-evidence/trader-gemini/core07-causal-pair-20261018-bdb20c69-89a6-4654-89f3-75ebeeac8291`. `MANIFEST.json` enumera originales, rutas de procedencia, tamaños y SHA-256; `VERIFICATION.json` registra la lectura independiente. Se conservaron exactamente un binario ya retenido por mundo, logs, recibos, wrapper/chain, bindings, corpus, parche, fuentes antes/después y limpieza auxiliar. No se copiaron target/cache, modelos binarios ni índices. La copia usa los retenidos históricos, nunca el ejecutable actual del target compartido.

## Identidad y resultados observados

| Evidencia | RED | GREEN |
|---|---|---|
| Commit privado | `e4705ec93366e616f51d6f1ceaecde52b3ce7a58` | `084dcaf4fa88733bf62705dd8c2411215f039cea` |
| Tree | `8e835a5878fb48a148fc8ef3c2d94820b4c24e31` | `25561b01ede699f7f3b29bbddd6fdcb643c19dd9` |
| Driver/validator | `command_failed`, 101/101 | `passed`, 0/0 |
| Wrapper/chain/coordinador | abortado 86, preservado | pasado 0 |
| Rust | 3 PASS, 2 FAIL, 0 ignorados/filtrados | 5 PASS, 0 FAIL, 0 ignorados/filtrados |
| Cargo UTC | 13:50:34.0616599–14:01:00.4050850 | 14:22:18.1177474–14:26:41.4332381 |
| Coordinador liberado UTC | 14:13:55.4225956 | 14:29:47.8911303 |

Comando idéntico: `cargo +nightly-2026-06-30 test --jobs 1 --release -p god-engine-core --test core_temporal_admission_contract --locked -- --nocapture --test-threads=1`.

Corpus SHA-256: `17e599ed1b37d8051a8644ef698dcd45a28710c634a7051240c8b362e41c6d54`. Su cabecera «ZERO Rust executions» pertenece a la propuesta histórica conservada sin reescritura; los recibos/logs posteriores documentan las ejecuciones reales.

En RED, el control positivo cerró una posición por SL con timestamp admitido 2001; la API FE rechazó 1999 con `BackwardTimestamp` y conservó sus campos del flujo. Aun así, Core cerró la posición y cambió el capital/reloj en las rutas **Dual y Depth**, causando los dos fallos de comportamiento. La prueba unificada falla en Depth antes de llegar al candidato negativo Trade: **RED no observó el fallo negativo de Trade**. Los tres controles positivos restantes pasaron.

En GREEN, los mismos cinco contratos pasan. La prueba unificada completa su bucle Depth→Trade, con control positivo de cierre previo a cada caso. También pasan igualdad BBO+Trade incluido origen 0, salidas defensivas con igualdad temporal y reloj anterior de otro activo. Esto prueba el contrato observado en las APIs y fixtures concretos; no afirma rechazo atómico de todos los errores FE ni cobertura exhaustiva de cada estado del sistema.

## Alcance causal y composición

La comparación de las dos fuentes archivadas identifica **sólo dos inserciones** en `crates/god-engine-core/src/lib.rs`, dentro de `process_event` y `process_tick_dual`: rechazar tiempo retrogrado por activo antes de mutar reloj/ledger, usando `tick_count > 0` y `event_time_ms < current_ts`. Igualdad y origen 0 permanecen admitidos. Preimage runtime SHA `c6db2d3c4aadd5c4cf74b5f30559d746746ad8f1a0d2701bd2b1693067981a94`; postimage `dcb3ce4bf36a6f8d66ecfd8f35a654bb064e7abcb7901e170bf45809e2577bc0`; parche `f9ae520adfbe6f63cdcd6ad693bd77987276f07baac60ba03e1815b59c33e4df`.

Esta composición pertenece al control privado LG5. Su commit `084dc…` **no es main público**. El puerto autónomo sobre el main vigente debe preservar cambios posteriores de Fable/otros agentes y ejecutar sus propios contratos antes de integración; no se publica todo LG5 por estos cinco PASS. HIGH01/02 y MED06 del expediente LG5 continúan abiertos hasta evidencia independiente.

## Coordinación y límite de recursos

La directiva original `OPERATOR_RESOURCE_DIRECTIVE.md` está copiada como dato, SHA `93d778ef2217fe2874ef5f7ee63dab957bd5a393cb18fa072c1cc8fa154d900e`. La orden del usuario es trabajar con la RAM disponible; ROOT fijó la admisión en **2 GiB físicos libres**, con `--jobs 1`, manteniendo target canónico, lock/heartbeat/Job y las guardas 12+8. El helper derivado archivado es `114cdfbd223f0e1063d8403cf2084c9cd4ef53a19669dd0e01fedd0cf5ae2f29`. Ambos coordinadores registran adquisición con inventario de compiladores vacío y RAM ≥2 GiB; esos cortes históricos no certifican aislamiento durante todo el trabajo ni ausencia de OOM.

RED registra explícitamente un Cargo debug AGY concurrente después de la admisión. No se declara aislamiento completo de RED. Los endpoints 12+8 tampoco certifican contenido/mtime del cache, todas las escrituras transitorias o todos los modelos. GREEN conserva su propia evidencia y no hereda un aislamiento que RED no tuvo. Los manifiestos originales de binarios contienen la frase histórica «isolated target»; `executed_path`, recibos y comando muestran que se usó el **target canónico compartido**, por lo que esa frase no se interpreta como un target aislado.

Ambos coordinadores liberaron el turno **después de limpieza explícita del auxiliar propio VCTIP**, no por drenaje espontáneo. En RED se verificó Job 1544/PID 64848 y el hijo de prueba ya terminal 58320/exit86; limpieza 14:13:51.8196700Z. En GREEN, el snapshot muestra Job 2356 con único miembro activo PID 42496 y el hijo 63456 ya terminal exit0 desde 14:26:45.4588777Z; limpieza 14:29:45.0613560Z y liberación 14:29:47.8911303Z. Los recibos B registran PID/StartTime, ruta, firma Microsoft y hash del auxiliar; no registran terminación de Cargo/rustc/tests/procesos ajenos. Se conservan también snapshots A. Según el operador ROOT, A se detuvo antes de matar: RED por precisión temporal CIM/native y GREEN por conversión automática DateTime de PowerShell; el archivo A no contiene resultado de limpieza. No se presentan esos intentos como PASS ni se alteran sus originales.

## Plan sincronizado al corte recibido

Restan seis pares privados: **Risk01** (RED real lanzado por ROOT después de Core07; sin resultado en este documento), Kelly02, Flow03, Hidden04, Replay05 y Reporter06. Cada RED requiere clasificación real antes de aplicar sólo su delta, y cada GREEN requiere misma fuente de corpus, binding/fuente nuevos y ventana propia. El orden actual lo controla ROOT; no hay ejecución automática por este archivo.

La observación posterior de Risk es otra operación. ROOT comunicó Cargo 57388 propio y Cargo adicional 43416 observado a 14:30:47Z, ausente al corte 14:32Z; después, 14:32:44Z, Cargo 36540/start14:32:01.9830360Z y rustc63180 además del ROOT57388, con RAM 2.10465 GiB. Se conserva como **informe del coordinador, actor no atribuido**, sin inventario nuevo por este auditor. No aporta aislamiento ni resultado funcional a Core07 o Risk.

T1 integral, HOST/R4, composición con main actual, cierre de hallazgos LG5 y publicación runtime siguen pendientes según sus propios gates. Este archivo es revisión documental de evidencia existente: **cero Cargo/rustc/CIM/Git calls, cero cambios en ROOT/índice/refs, cero permisos de merge concedidos** por el auditor.

## Pines de los originales principales

| Original | SHA-256 |
|---|---|
| RED driver | `621af3e8bea096664bc5ef60e9dee935d4b161b66c2f11f4fb1594162474d0b6` |
| RED log | `f8b5efcc6e5e3abeca93718a30843c87e3d3f4b74455df76957e4a5ad48a58e4` |
| RED coordinador | `4c72aa11cc45acdd4bee7822ad1477721c797fd333d00b695c23a01b843c5de1` |
| RED binario retenido | `e4dc8a0deab97b4718ed1002e370d8c81db91de4f1a3bd2e4b027077d441cd69` |
| GREEN driver | `bd382b3a9734a93aeb8dc74b6162e8476a4361bd4c581eaa190c5063e78042f5` |
| GREEN log | `d937caf9192e955ea50c79bddab148247dea592b4b3678fa5dd55ba0a6ee3826` |
| GREEN coordinador | `963c0537b3920df15f4795439e285299ea797e6bc8fa271c54644e34a1a9656c` |
| GREEN binario retenido | `d2250eea4122376b3b2c8fc0165cae7894015d2f20813505299307dcb3d047cc` |

Revisión documental ROOT para esta publicación: borrador original SHA256 `01aee484de9311445c948d7b94bccd3e3c7f226016980d9b7507b3ec7aa44b54`; manifiesto del archivo `15410d8e06efb2d750337769e9a7da5175279a812de8bd76f6e5e8bfc9692b55`; verificación independiente `2af34c10fba145f2171a05ff8882adae77ba432c57fd3a9f9ef45bb94dd5cceb`. Se precisó la severidad MED06 y se distinguió la orden de trabajar con RAM disponible del umbral operativo elegido por ROOT. Los originales archivados permanecen intactos.
