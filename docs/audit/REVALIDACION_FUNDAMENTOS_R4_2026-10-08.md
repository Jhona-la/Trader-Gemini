# R4 — Revalidación de fundamentos, 2026-10-08

Estado: revisión estática de fuentes fijadas y un testigo Rust aislado expresamente autorizado. No Cargo, T1, datos privados, evaluación económica ni cambios a fuentes, índice o HEAD. El informe se preparó como salida local ignorada. Su publicación y la equivalencia de fuentes posteriores tienen recibos separados; esta revisión no acredita despliegue.

## 1. Procedencia y alcance

- Origen remoto refrescado por el integrador: `18bbd1d90a073ae49ca8083c5898537013e4bfeb` (Ω25).
- Padre local de integración: `f3f86960190d99aeacdb75b14cdfab0adceaa186`; no se presenta como origin/main actual.
- Árbol candidato inicialmente leído: `2777578104f7931bb44f7e053103026b45ca811c`.
- Árbol candidato posterior: `f26f3b83e784dac821b53a5363f19bb29baf3c4d`. Comparación entre esos dos árboles: sólo `crates/god-engine-core/tests/stateful_transition_contract.rs`; las fuentes de esta revisión permanecen idénticas. Ese árbol quedó fijado después por el commit local `7acf36aadf33a323b368f8e59fb77e2aafd40720`; no se presume publicación remota.
- Worktree del informe: `quant-foundations-2026-10-07/Trader Gemini`, HEAD `639c3e0d7c5f0dc8c376ac04af2a2fdc7a5bc5a9`. Las lecturas actuales usan objetos Git explícitos; no confunden ese checkout antiguo con la fuente integrada.
- La ausencia de recuperaciones OOS/SA/model reload/publicación en origin18bb respecto de f3 describe dos líneas de trabajo pendientes de unión; no demuestra que Ω17–Ω25 borraran esas recuperaciones.

## 2. Revalidación Q2, Q3 y Q4

| Expediente | Estado en origin18bb y candidato | Alcance demostrado y siguiente contrato |
|---|---|---|
| Q2, alto condicional | Persiste capital realizado y drawdown de cierres. El comentario de `darwin.rs:352` ahora lo reconoce, pero añade una afirmación no demostrada de DSR conservador. | `evaluate_genotype` lee `unified_capital` y actualiza pico/DD al cerrar, sin sumar PnL abierto. El core conserva contabilidad separada: entrada resta comisión, cierre acredita PnL neto y `pnl_unrealized` vive aparte. Falta curva de patrimonio MTM común, reconciliación de flujos/costes y ventana económica. No prueba de drawdown vivo ejecutada aquí. |
| Q3, medio condicional | Corregido el disparador extra por cierre y el reseteo de fase. No queda demostrada una serie de retornos de duración uniforme. | `darwin.rs:387–398` hace un solo `push` por tick que cruza la frontera; después avanza el reloj con `while`. Ticks en 0,1000,4500,5000 ms emiten tres retornos de duraciones 1000,3500,500 ms, frente a cinco intervalos nominales de 1 s. El vector no guarda tiempos ni máscara de huecos. Es un contraejemplo estático, no una corrida del motor. Definir política causal de muestreo/faltantes y comprobar con gaps; no rellenar precios desconocidos retrospectivamente. |
| Q4, alto para inferencia de confianza | Floor de curtosis a 3 y fallback de skew nuevos; dependencia temporal, contaminación y dispersión entre ensayos siguen abiertas. | `selection_stats.rs:29–51,83–107,139–154,228–252` sólo emplea momentos marginales, n y número de ensayos; no hay HAC/autocovarianza, timestamps ni identidad de observación. El testigo actual de §3 reproduce aprobación por duplicación y aceptación de serie contaminada. No demuestra una tasa de error del bot ni invalida por sí solo todas las aplicaciones i.i.d. |

### Lectura de fórmula, fallback, mutabilidad y datos

1. La fórmula implementada conserva `−skewness * SR`. Por ello, alta curtosis no implica universalmente mayor error estándar que `1/sqrt(n−1)`. El testigo de §3 refuta ese comentario con momentos calculados por el mismo módulo; tampoco certifica que el score alto resultante tenga cobertura nominal.
2. El fallback nuevo elimina skew sólo cuando el discriminante no es positivo. Su test unitario construye momentos manuales `skew=2,kurtosis=3` y un SR independiente de media/sd: ese ejemplo no es evidencia de una distribución poblacional coherente, pues viola `kurtosis >= skewness²+1`. Prueba una rama defensiva sobre la API pública; no valida calibración, dependencia ni datos producidos por `compute_moments`.
3. `compute_moments` elimina NaN/Inf sin devolver su número ni un estado de contaminación. Al menos 20 valores finitos bastan para seguir, y `SelectionVerdict` no transmite pérdidas de observaciones. El contrato del consumidor debe decidir explícitamente qué pérdidas tolera; no se ha probado que el feed vivo produzca la serie del testigo.
4. `dsr` usa el error estándar del Sharpe del candidato como dispersión para el máximo entre ensayos. Esa sustitución requiere un modelo de la familia de ensayos; el contador por sí solo no lo suministra. La formulación DSR distingue selección entre ensayos y error de estimación del ganador; revisar conjuntamente N efectivo y dispersión/dependencia entre candidatos. [Bailey y López de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
5. El módulo de momentos es determinista sobre un slice, pero Darwin no congela una identidad efectiva conjunta de arena/modelos/datos: usa replay con macros sintéticas y modelos globales cargables. El comentario de `darwin.rs:329–332` reconoce ese límite. Separar reglas adaptativas preregistradas de mutación externa no registrada; un snapshot inicial tampoco hace independiente una consulta OOS repetida.
6. Q1: origin18bb aún reconstruye daemon por ronda en el host; el candidato conserva el `Arc<DarwinDaemon>` de f3 y clona el Arc en cada worker. El contador persiste dentro de ese proceso secuencial; no persiste tras reinicio ni convierte el DSR en prueba válida para parada opcional. No asumir ausencia de races para callers adicionales concurrentes de `evolve_online` sin auditar esa interfaz.

## 3. Testigo Rust actual y reproducible

Directorio: `target/r4-review/selection_stats_18bb_witness/`.

- Fuente: copia binaria exacta del blob `1e2ba9b194cf133bc614071c168003287558145d`, igual en origin18bb y candidato. `git hash-object` de la copia coincide.
- SHA256 fuente: `a1995b82b66671b5083397e4e96a466020e13aadf32ec836808c868ffc9b88d6`.
- SHA256 wrapper `witness.rs`: `5ea3d91d4bc9bd79ba117d8c10b5c5db95123f6961095f9e668a169d34c0e86f`.
- Compilador `rustc 1.98.0-nightly (096694416 2026-06-29)`, `x86_64-pc-windows-msvc`.
- Comando conceptual reproducible desde el worktree: `rustc --edition=2021 target/r4-review/selection_stats_18bb_witness/witness.rs -o target/r4-review/selection_stats_18bb_witness/witness.exe`; ejecutar ese exe. `receipt.json` guarda argumentos absolutos exactos, versión, hashes y stdout.
- Exit de compilación 0 y ejecución 0; logs `compile.log` y `run.log`. Compila el módulo original como módulo Rust; no es port Python. No ejecuta sus unit tests, el pipeline ni T1.

| Entrada del testigo; N ensayos=100 | n que usa el módulo | DSR | pasa |
|---|---:|---:|---|
| `[0.0026,-0.0014]` repetido 20 veces | 40 | 0.235773213910289 | false |
| Cada observación anterior copiada consecutivamente 10 veces | 400 | 0.999556946880623 | true |
| Esas 400 observaciones + NaN,+Inf,−Inf (403 entradas) | 400 | 0.999556946880623 | true |
| Las 400 observaciones ordenadas | 400 | 0.999556946880623 | true |

Copiar datos no añade innovaciones independientes. La sensibilidad a n y la invariancia al orden exhiben falta de un contrato de dependencia/deduplicación; la API no puede inferir duplicados económicos sólo a partir de valores repetidos. No es legítimo extrapolar estos números a error real de selección en producción. Los valores históricos del port Python se conservan como históricos y no se sustituyen silenciosamente por estos resultados.

Contraejemplo adicional a «curtosis alta ⇒ SE mayor»: 40 valores, un spike de 0.01 y desplazamiento común para SR=0.5, generan `kurtosis=36.148125`, `skew=5.858119365461953`, `SE=0.082849590279387`, inferior a `1/sqrt(39)=0.160128153805087`. DSR=0.999771163076773. La comparación es contra el baseline explícito del comentario, no una afirmación general de cuál es el estimador correcto bajo esos datos.

## 4. Residuales I1–I4

| Expediente | origin18bb | Candidato posterior | Límite/prioridad |
|---|---|---|---|
| I1, tiempo AggTrade | Persiste `extract_u64_from(...T...).map(...).unwrap_or(0)`, parser:116–118; `update_agg_trade`, state:715–735, suma cantidad con decay=1 si tiempo no avanza. | Parser difiere por recuperación de contratos BookTicker, pero el fallback AggTrade sigue igual. | Separar campo ausente de inválido, duplicado y reloj retrógrado. No reproducción del host ni prueba de payloads vivos aquí. |
| I2, anclas SA frente curvas | `src/bin/evolution.rs` muta TP/SL legacy y no llama a reconstruir curvas antes de evaluar; `SuperGenotype` hace autoritativas las curvas. | Recupera OOS y SA, pero mutaciones de anclas siguen en 267–268/301–302 y no aparece `rebuild_tp_sl_curves_from_anchors`. | La corrección Ω16 del otro binario continuo no cierra este consumidor. Oráculo mínimo futuro: un cambio de TP/SL propuesto debe cambiar los valores efectivos del arena/replay y su identidad. |
| I3, modelos | Host usa mtime y marca versión antes de éxito; no incluye módulo `model_reload`. | Recupera un tracker común con selección JSON/bin y registro sólo al éxito. | Mejora concreta conservada. Pendiente identidad de contenido/cache ligada a fuente, revocación de modelo ausente, misma metadata con contenido distinto; DarkAlpha sigue watcher separado y el loader es síncrono. No se afirma que esos fallos ocurrieron en vivo. |
| I4, familia rho_tau | Renombra constante `ESCALAS_BANDA_PAR=5`; no cambia dominio real ni proceso por escala. | Misma matriz/escalas, y publicación recuperada con ausencia explícita. Callers actuales `lib.rs:1667,2749,3764`. | El conteo por par sigue incompleto como descripción de dominio; subcobertura global del host de 26 no demostrada. No alterar política/umbrales en esta revisión. |

I4 en detalle: el selector por distancia absoluta a toda la malla permite índices 17…22 para tau en [30 s,12 h]. En 30 s selecciona k17=17179.869184 ms; el fallback dominante puede llegar a 30 s. Hay almacenamiento y actualización de e-proceso en k17; las guardas no lo excluyen por pertenecer fuera de 18…22. La consulta valida índice<32, madurez, dispersión y evidencia; la recencia se aplica al emparejar bloques, no es una nueva medición de antigüedad al consultar. Esta inspección no ejecutó el fixture de maduración.

`C(26,2)*6=1950 <2175`: el roster configurado de 26 no demuestra insuficiencia de la reserva global. Para capacidad 30, `C(30,2)*6=2610 >2175`, lo que exige un contrato explícito antes de afirmar cobertura para capacidad completa; no prueba una tasa real del 6% ni recepción efectiva de los 30 activos. Mantener aparte 32/640 de escala univariante, 416/8320 de motores y 2175/43500 del veto. La validez depende además del nulo condicional y filtración del e-proceso; un nombre o umbral no la certifica. [Ramdas, Grünwald, Vovk y Shafer, SAVI](https://arxiv.org/abs/2210.01948).

## 5. Preservación semántica acotada del merge

Comparación contra ambos padres, no sólo ausencia de conflictos:

- Darwin conserva el evaluador de18bb, renombres de anclas y nuevo reloj; el host conserva Arc persistente de f3 y cambio de SL fallback a curva de18bb.
- `selection_stats` del candidato es idéntico a18bb; `evolution-engine/src/lib.rs` mantiene re-export canónico de risk y añade los módulos SA recuperados. No aparece una implementación HAC nueva en estas fuentes.
- `risk-engine/src/lib.rs` conserva el guard de no finitud antes de mezclar Kelly y `clamp_ruin`; `ruin.rs` no fue reemplazado.
- `orchestrator.rs` permanece igual a f3: valida presión sistémica y pares espectrales no finitos, reusa valores leídos y conserva contracción continua Ω12 y veto sistémico >=0.90. No restaura la antigua conversión de presión desconocida a calma.
- No detectada pérdida de esos contratos por integración. Es aprobación limitada a preservación de fuentes examinadas; check21/T1/CI y SHA publicado son recibos del integrador, no resultados ejecutados aquí.

## 6. Revisión de borradores científicos y siguiente lote

Revisados los cuatro borradores locales bajo `target/r4-review`: PROTOCOLO_OOS, RESIDUALES_INTEGRACION, PLAN_MAESTRO_QUANT_SR y PLAN_MAESTRO_SINCRONIZACION.

Correcciones concretas recomendadas al editor:

1. PROTOCOLO_OOS:55 exige que «la estimación existía antes del intervalo evaluado». Para reconstrucción histórica eso confunde fecha material del artefacto con causalidad informativa: usar «estimada exclusivamente con información disponible antes de cada decisión evaluada», con disponibilidad de etiquetas/versiones y genealogía. Un examen prospectivo sí debe fijar previamente el protocolo; no usar esta precisión para justificar ajuste tras observar OOS.
2. PROTOCOLO_OOS:94, «se ejecuta una sola vez», es demasiado absoluto para reproducibilidad y semillas/réplicas preregistradas. Prohibir cambios informados por el resultado y registrar toda consulta; reejecución idéntica no aporta nueva evidencia independiente.
3. PROTOCOLO_OOS:88 dice política adaptativa congelada; §6 aclara correctamente congelar reglas de actualización, información permitida y estado inicial, pudiendo actualizar causalmente parámetros. Uniformar esa expresión en la tabla para que no parezca prohibir la adaptación que luego se permite.
4. Mantener citas a639/f3 como historia y añadir alcance18bb/candidato final. Q3 ya no debe describirse simplemente como disparo cierre+1s vigente; ahora el residual son gaps y retornos sin duración. Q4 debe citar los resultados Rust actuales y mantener los valores Python antiguos como evidencia del estado anterior.

Sin objeción a las correcciones de 26 configurados/30 capacidad, dominios dimensionales por productor, malla representable frente resolución del reloj, autores SAVI, GBM/Kelly idealizado frente meta económica o separación T1/paridad/retorno. I4 ya distingue correctamente conteo y cobertura; evitar volver a afirmar subcobertura viva universal.

Recibos aún necesarios: R1 identidad temporal, metadatos/hash de datasets y labels maduros; R2 estimando 72 h, dependencia y genealogía de ensayos; R5 patrimonio MTM, flujos/costes y frontera económica tras warm-up; R6 identidad efectiva de todos los modelos/genomas, protocolo de adaptación, promoción/reversión y custodio de OOS. Ningún cambio fuente examinado acredita por sí solo un examen sellado ni crecimiento de 100%/72 h. Prioridad siguiente: corregir primero contratos de medida y procedencia, luego evaluar teorías sobre una base comparable.

## 7. Blobs fijados

`ausente` significa ruta no incluida en ese árbol; no demuestra borrado por la rama remota.

| Ruta | origin18bb | candidato f26f3b83 |
|---|---|---|
| `crates/god-engine-core/src/darwin.rs` | `847a6e96e9b4de13cb0d473c25e159662a7e6cf0` | `847a6e96e9b4de13cb0d473c25e159662a7e6cf0` |
| `crates/risk-engine/src/selection_stats.rs` | `1e2ba9b194cf133bc614071c168003287558145d` | `1e2ba9b194cf133bc614071c168003287558145d` |
| `crates/data-pipeline/src/parser.rs` | `f07fd917a23ecbef5a7ea557233df9aa89d9cc72` | `6c47744a94f5bea150e8a65c750b2a2f58c73cc4` |
| `crates/data-pipeline/src/ws_client.rs` | `7e9ec19d593828e9eb76bb12b5c5dc4335f50c35` | `7e9ec19d593828e9eb76bb12b5c5dc4335f50c35` |
| `crates/quantum-arena/src/state.rs` | `a9b0813d22d967458416c2e5b8c050cdb014a211` | `a9b0813d22d967458416c2e5b8c050cdb014a211` |
| `src/bin/evolution.rs` | `44d7d0f58a4cea8efcaf1cc788ad69094a47c7b9` | `30f34f1b205edf7929fbcfa8ace181d5a546e1d4` |
| `crates/quantum-arena/src/genome.rs` | `ca35f0cc12bf6e5031b6de65eb146623e5e8905c` | `ca35f0cc12bf6e5031b6de65eb146623e5e8905c` |
| `src/bin/god_engine.rs` | `5abe2c589bd15f7cd4cecfddeb2eec9f338b6042` | `eea831560cce74f00570b230724f5eef3f106420` |
| `crates/god-engine-core/src/ml_inference.rs` | `6c771d38b8aee002c876c5efa1f76c582db9e8ee` | `6c771d38b8aee002c876c5efa1f76c582db9e8ee` |
| `crates/god-engine-core/src/model_reload.rs` | `ausente` | `3357a6d870b5c7c6176958faa793324e0eec45f2` |
| `crates/quantum-arena/src/espectral_multiactivo.rs` | `a480b0d2cc3449fc07ca569dbd0606af84fcbf8c` | `a480b0d2cc3449fc07ca569dbd0606af84fcbf8c` |
| `crates/quantum-arena/src/evalues.rs` | `758ff0af9725b73ffdbb054e84e3e025c0435a94` | `758ff0af9725b73ffdbb054e84e3e025c0435a94` |
| `crates/quantum-arena/src/temporal_spectrum.rs` | `45c2bba7f8af60b3f160b1327de8468d8c094013` | `45c2bba7f8af60b3f160b1327de8468d8c094013` |
| `crates/god-engine-core/src/evidence_publication.rs` | `ausente` | `b0d34fe01fe4e08476ed05e8def2996fc4b3d30b` |
| `crates/god-engine-core/src/lib.rs` | `869d166b60c9b98e00e7c05fa2f4e0dd5b3d9d57` | `b045efbf82766090478c9da17412b05acb7d2810` |
| `crates/god-engine-core/src/bootloader.rs` | `6ae4163dbef146f40ecf8b1633417e1f6dfe76c1` | `6ae4163dbef146f40ecf8b1633417e1f6dfe76c1` |
| `crates/risk-engine/src/lib.rs` | `35022083e831b8813d56d82b34041c232ef4abf2` | `554e02d060059d9d15247a2194df6d65f0f3ea36` |
| `crates/risk-engine/src/ruin.rs` | `4952f122397efe9a23e35aac2ad05adaf137a4e0` | `4952f122397efe9a23e35aac2ad05adaf137a4e0` |
| `crates/risk-engine/src/orchestrator.rs` | `a30856f4e03f1d989bb7c593b522b4017b8f9626` | `3ee9650476fd1ab867a780cef6f01f3f2e3fcafc` |
| `crates/evolution-engine/src/lib.rs` | `2d3ed5c22607826a0ae701c2f36f7098aa0a07bf` | `6c60560c2be84f6e11e9978eca821b7a0b2893ce` |

## Reproducción durable de los diagnósticos

El paquete `scripts/audit_r4_rust_witnesses.py` recupera cuatro blobs
inmutables de `18bbd1d9`, verifica su identidad y usa los wrappers
versionados en `scripts/fixtures/`. La extracción SDE conserva
exactamente líneas 197–337, sin adaptar fórmulas.
El [recibo combinado](RECIBOS_DIAGNOSTICOS_RUST_R4_2026-10-08.json)
registra cinco comandos exit 0, hashes y salidas sintéticas. El éxito
caracteriza los defectos; no ejecuta el workspace, OOS ni operación.


## Corte posterior de código integrado

El commit `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`, árbol `10b8a427dfbdd5b203595b039b2dad50c69d6f68`, incorpora main `5842c8e33231131e5f2c5c600ffce15f1babaa55`.
La comparación completa frente a7acf cambia sólo memoria/coordinación/
triaje, el binario continuous_evolution_backtest (retira dos overrides
de latencia y un import) y el nombre de un test stateful_open.
Los blobs científicos, core/OU/registry, guard C07 y consumidores
de los ocho hallazgos anotados permanecen idénticos. La tabla de
14 rutas OOS conserva su corte histórico: el binario continuo
posterior es blob `e3d29c736be66f5c18baaf2f78d577f88bdef7b9`;
su cambio no cierra los recibos económicos pendientes. Los resultados
Cargo/T1 y su alcance quedan en el [recibo de integración](../INTEGRACION_RAMAS_2026-10-07.md).
