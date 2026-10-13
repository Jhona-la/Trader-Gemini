# R4 — cierre de T1 histórico y reconciliación de integración

Fecha del cliente: 2026-10-11. Auditor: Codex, revisión independiente de
recibos, objetos Git, refs y estados de worktrees. Este documento registra
observaciones; no declara una publicación, limpieza o nueva prueba realizada
por el mero hecho de escribirlas.

## Resultado comprobado de T1 del 8 de octubre

T1 terminó correctamente: **16/144 genes sensibles (11,111 %), umbral 0,110
intacto, 1 prueba aprobada, 0 fallidas, 1 filtrada**. La ejecución terminó en
`2026-10-08T21:06:01.763625+00:00`. El comando completo consumió 1619,667 s;
el resumen del ejecutable registra 1481,02 s de prueba después de compilar.

El recibo `48-omega28-t1.json` fija antes y después el commit
`eb32ac9613c847eec75bd8e2591add8ae88234cf` y el árbol
`10ecee6c773a37d6ac6a0ff7908f3c2c800d8ea0`. El SHA256 verificado del log
`48-omega28-t1.log` es
`864c7daf715b4536288d06287e07b31e25933ac59843a26d8ca44a9fd41cef68`.
El log muestra recompilación de `strategy-core`, `signal-engine`,
`risk-engine`, `execution-engine`, `god-engine-core` y `backtest-engine`
desde el worktree `integration-recovery`.

```text
cargo +nightly test --release -p backtest-engine --locked \
  --test t1_cobertura_genetica \
  t1_cobertura_genetica_del_oraculo_de_aptitud \
  -- --exact --nocapture --test-threads=1
```

El ambiente registrado fue `TG_GENOME_ENV=backtest`,
`CARGO_BUILD_JOBS=1` y `CARGO_TARGET_DIR` en el `target` del checkout
principal. El recibo de composición registra Rust
`1.98.0-nightly (096694416 2026-06-29)`, Windows MSVC, LLVM 22.1.8.
La prueba histórica no guardó una huella del ejecutable: los registros de
recompilación y hashes son evidencia de ejecución, pero no sustituyen la
separación de artefactos que exige una nueva comparación entre worktrees.

**Alcance:** sensibilidad de una estadística de replay a un extremo por
coordenada, condicionada al fixture sintético. No demuestra rentabilidad,
fitness canónico, suficiencia estadística, inercia global de genes ni el
objetivo de duplicación cada 72 horas. Cinco perturbaciones no produjeron
cambio efectivo de coordenada: 13, 14, 15, 16 y 140. Tampoco acredita el
código actual mediante identidad global de fuentes.

## Regresiones de la misma composición

Se releyeron los siete JSON de ejecuciones 42–48 y se recalculó el SHA256 de
cada log. Todos coinciden, tienen exit 0 y conservan HEAD y árbol anteriores
y posteriores de `eb32ac96`/`10ecee6c`.

| Recibo | Aprobadas | Fallidas | Ignoradas |
|---|---:|---:|---:|
| 42, biblioteca strategy-core | 29 | 0 | 0 |
| 43, cuatro contratos strategy-core | 12 | 0 | 0 |
| 44, biblioteca god-engine-core | 170 | 0 | 0 |
| 45, diez contratos god-engine-core | 115 | 0 | 1 |
| 46, stateful_open_diagnostics | 5 | 0 | 0 |
| 47, reporting_contract | 2 | 0 | 0 |
| 48, T1 exacto | 1 | 0 | 0 |
| **Total único** | **334** | **0** | **1** |

El RED esperado de la prueba del consumidor y su GREEN repetido quedan
fuera de esta suma. Tampoco se añaden las pruebas históricas de `e9c6a435`
ni de `7acf36aa`. Los contratos que describen deuda abierta no la reparan.
El campo `state` original de `omega28-composition.json` sigue siendo
`Completed final validation; publication pending`; no se ha reescrito para
aparentar que su publicación ocurrió.

Recibos y logs originales conservados en:

```text
C:/Users/jhona/.codex/worktrees/integration-recovery/Trader Gemini/
  target/integration-receipts/2026-10-08-refresh/
```

## Qué llegó a main y qué quedó pendiente

En el censo de esta revisión, el remoto real `refs/heads/main` era
`1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a`. Al iniciar, el checkout y su
tracking todavía eran `c4e8e2856a18e32e81027bc546c6c41544e85c4d`.
La integración principal actualiza ese tracking; no se presupone que el
remoto deje de avanzar durante el trabajo concurrente.

La rama histórica `codex/integration-recovery-2026-10-07`, HEAD
`eb32ac9613c847eec75bd8e2591add8ae88234cf`, **no es ancestro** de ese main:
conserva tres commits exclusivos por identidad (`4feb7356`, `e584d3ef`,
`eb32ac96`). Esto no significa que el arreglo OU siga ausente.

El commit integrado
`325008aecd2d9c83272b165016af17635aa3cb7a` (Ω34) incorporó la condición
`sde.count > 0 && timestamp_ms <= sde.last_ts_ms` antes de producir una
nueva intención. El contrato
`crates/strategy-core/tests/omega28_consumer_clock_contract.rs` tiene el
**mismo blob** en `eb32ac96` y en `origin/main`:
`f8812c55c5a5c2fbfb1152576c7e5886b5eea320`.

La implementación completa del motor OU ha cambiado desde entonces:
amplió la vida media elegible a 12 horas, cambió su curva de confianza y
corregió el uso de la media previa en el regresor discreto. Por tanto, no
se debe reemplazar `multivariate_coint.rs` con su versión del 8 de octubre
ni reutilizar T1 histórico para certificar estos cambios posteriores.

Los tres documentos exclusivos de publicación de `4feb7356` todavía no
existían en el main del censo:

- `docs/audit/DELTA_CIERRE_2026-10-08.json`.
- `docs/audit/RECIBO_PUBLICACION_R4_2026-10-08.json`.
- `docs/audit/REVISION_CIERRE_2026-10-08.json`.

También quedaban sin incorporar las adendas históricas correspondientes
en memoria, coordinación y cinco documentos de planes/integración. Sus
conteos, reservas y SHA describen aquel corte del 8 de octubre; una nueva
adenda debe distinguirlos del estado de hoy.

**Integración recomendada:** merge histórico con unión exacta de esas
adendas y documentos, preservando el runtime actual y sus cambios
posteriores. Comparar el resultado contra ambos padres y ejecutar
`cargo check --workspace --all-targets --locked` antes del commit de merge.
Después ejecutar los contratos pertinentes y T1 sobre la composición
actual, publicar mediante push normal y comprobar el OID remoto. Sólo
entonces liberar la ref histórica; conservar el directorio y sus recibos.

## Censo de ramas y reservas observado

| Ref | OID observado | Exclusivos frente a main remoto | Estado y decisión |
|---|---|---:|---|
| codex/integration-recovery-2026-10-07 | eb32ac9613c847eec75bd8e2591add8ae88234cf | 3 | Limpia; integrar y publicar antes de liberar. |
| qoder/r7r4-nucleo-vivo | ce5e08979f9be9d6e479fbc16e429013c974f91b | 0 | Limpia, fase cerrada y SHA integrado; candidata a liberar conservando `.r7r4` al mismo HEAD. |
| qoder/ronda7-plan | cc4ce8f5a63d3b921b11a91ed5f7e1523b2b4990 | 0 | Limpia, fase cerrada según plan; candidata a liberar conservando `.ola73` al mismo HEAD. |
| qoder/r7r6-aprender-medir | 6376d089dd266ea87d1003b403d13e6498d375a0 | 9 | Reserva activa y cinco documentos modificados; conservar. |
| sol/replay-accounting-2026-10-09 | f5cadac7023da4fb9e658f02c6bffd8606135f3f | 0 | Limpia pero reserva activa de replay/metrics; conservar. |
| origin/claude/auditoria-deslizamiento-apalancamiento-sqtc08 | 8147c71f7085f76ce3eaa871bf0022ceec842a01 | 1 | Nuevo aviso C-F2 sólo tests, posterior al PR #38; conservar. |
| origin/claude/elegant-euler-mmtht4 | f1a70192723261172553cb22d2e96814ecbac720 | 0 | PR #36 fusionado; candidata remota sólo tras excluir nueva actividad y confirmar el mismo OID. |

GitHub confirmó PR #38 fusionado en `2026-10-11T10:49:10Z`, PR #37 en
`08:21:54Z` y PR #36 en `04:54:35Z`; no había PR abierto en el censo.
La rama C está reutilizada y publicó una nueva asignación aunque su PR
anterior ya esté fusionado. Una PR cerrada por sí sola no autoriza borrar
trabajo nuevo. La rama E conserva su dominio de investigación: antes de
eliminar su ref debe comprobarse que no inició otro ciclo.

La limpieza candidata exige OID fresco, ancestría en main publicado,
ausencia de exclusivos, reserva cerrada y preservación de status, índice
y archivos del worktree antes/después de un detach al mismo HEAD. No
elimina directorios, cachés, recibos, modelos ni trabajo ajeno. Esta
auditoría no ejecutó ninguna mutación Git ni compilación.

## Adenda de integración y limpieza — 2026-10-11, segundo censo

Este corte sustituye las decisiones candidatas del primer censo; conserva
sus observaciones como historial. El remoto `main` continúa en
`1d1f8bdd35640e5ab12b6e55edb9bf582c41c33a`.

- Las refs locales `qoder/r7r4-nucleo-vivo` y `qoder/ronda7-plan` ya no
  existen. Los directorios `.r7r4` y `.ola73` permanecen en detached HEAD,
  respectivamente `ce5e08979f9be9d6e479fbc16e429013c974f91b` y
  `cc4ce8f5a63d3b921b11a91ed5f7e1523b2b4990`. Este segundo censo comprueba
  ausencia de refs y conservación de directorios/HEAD; la comparación de
  bytes antes/después pertenece al recibo de limpieza del integrador.
- Claude E se **conserva por reserva activa**: tomó la versión 2 del
  fixture T2 y AGY acusó esa asignación. Su ref remota sigue en
  `f1a70192723261172553cb22d2e96814ecbac720`; ser ancestro y tener PR #36
  fusionado no cierra su nuevo ciclo ni su dominio D1/D4.
- Claude C avanzó a `1f752037bafca6817e2551eb3d21fd87882fb27b`, tanto en
  local como remoto. Su worktree `.claude/worktrees/determined-tu-0d7126`
  está activo y tiene siete tests modificados. Esos cambios no se deben
  integrar ni borrar anticipando el cierre de su dueño.
- SI publicó `codex/spectral-evidence-1011` en
  `06ce88efd3fd05638eea2b7cdf0698dcf235e066`: commits `8404bbdf`,
  `c9d877da` y `06ce88ef` desde `1d1f8bdd`. Cambian `spectral_tape.rs`,
  `state.rs`, `spectral_bench.rs`, tests y `unit-suites.yml`. Además, su
  worktree tiene tres documentos modificados y dos nuevos. Los resultados
  del lote ceros/replay no se heredan a esta composición diferente: al
  integrar SI habrá que revisar ambos padres y probar las rutas afectadas
  sobre una nueva fuente congelada.
- El checkout compartido está ahora en la rama activa
  `antigravity/ola83-revision-base-c-f1-f002-rigor-dimensional-unidades`,
  HEAD `1d1f8bdd`. La ref `main` sigue en ese OID, pero el checkout ya no
  pertenece a `main`. No cambiarlo ni forzar un fast-forward allí sin
  revalidar rama, estado y reserva. Qoder R6 mantiene sus cinco documentos
  modificados y nueve exclusivos; Sol mantiene su reserva de replay.

### Resolución exacta del merge histórico propuesto

La base común del merge con `eb32ac96` sigue siendo
`a5c69af3f7a05c8642b803c61a6ff72613d96100`. El preview comunicado por el
integrador presenta conflictos sólo en memoria, coordinación y
`crates/strategy-core/src/multivariate_coint.rs`. Este informe no ejecutó
un merge ni modificó sus archivos congelados.

La resolución debe comenzar después de cerrar el commit de ceros/replay.
No aplicar de nuevo el parche viejo del consumidor OU. El blob del motor
en la base actual es **`421e4277546c5ca45b79a37f6cb403c796d941d8`**; debe
quedar idéntico tras el merge histórico. El contrato OU debe seguir siendo
**`f8812c55c5a5c2fbfb1152576c7e5886b5eea320`**. Si otra sesión cambia
estos objetos antes de integrar, parar la resolución automática y revisar
ese delta; no restaurar los OIDs anteriores por inercia.

Se verificó independientemente una forma exacta de los dos documentos
conflictivos en los objetos Git: `incoming = H + D + B`, y
`base = H + B`. `H` es el título y la línea vacía, `D` la inserción de
cierre D2, y `B` el resto completo de la base. Los hashes del encabezado
y del resto coinciden; la inserción no sustituye ninguna línea previa.

| Documento | Bytes H | Bytes D | SHA256 del inserto D, bytes Git LF |
|---|---:|---:|---|
| `.agents/MEMORIA.md` | 56 | 1272 | `5c69fe83bb79f38e269e816e860ff32ce82926c7c3a1b8f1063d6ad99e9d9443` |
| `COORDINACION_CODEX_2026-09-28.md` | 53 | 1263 | `f73c8bacf9ed77a72eb0fca77216a50e53193591b1ddc4fa90eed31f9df6c289` |

Para unirlos, tomar el documento **completo del padre actual** y añadir
ese inserto histórico exacto como una sección del 8 de octubre. Una forma
que preserva todo el cuerpo actual sin reconstruirlo es anexar `D` al
EOF; comprobar primero que termina en LF. También puede insertarse tras
`H` si se comprueba que el resto actual permanece idéntico. No elegir
todo el archivo de uno de los padres, eliminar entradas posteriores ni
copiar bloques ya presentes dos veces. La nueva adenda vigente del
integrador debe aclarar que las reservas y pendientes de D2 son
históricos.

En las otras cinco rutas documentales de D2, preservar los nueve renglones
añadidos por `4feb7356` y todos los cambios posteriores del padre actual.
Tres de esas rutas no cambiaron desde la base común; las dos con cambios
posteriores necesitan comparación de contenido aunque el merge no marque
conflicto. Verificar el resultado contra ambos padres y comprobar que el
delta frente al padre actual sólo añade los metadatos históricos previstos
en este merge. Los inputs compilables del lote actual deben permanecer
idénticos; los archivos ceros/replay del nuevo commit no se reemplazan.
Ejecutar el check de workspace con `--all-targets --locked` antes de
commitear el merge, y mantener los resultados de T1/regresiones vinculados
a su SHA/árbol de ejecución o a una equivalencia completa documentada.

### Los tres JSON D2 recuperados por el merge

Son adiciones históricas independientes de los siete documentos copiados
recientemente. Sus blobs en `eb32ac96`, que el merge debe preservar, son:

| Archivo en `docs/audit/` | Blob Git histórico | Qué acredita |
|---|---|---|
| `DELTA_CIERRE_2026-10-08.json` | `aefe3183a9674cdbe28444c51a27004fda24beb2` | Delta del snapshot publicado del 8 de octubre. |
| `RECIBO_PUBLICACION_R4_2026-10-08.json` | `3c356e36a09dc64c8dee0896c469d7d269390258` | Push D1 `a0a5b982`, runtime `e9c6a435`, 532 inputs idénticos y limpieza de aquel corte. |
| `REVISION_CIERRE_2026-10-08.json` | `01b735ac66b59093aaafd541e195a55802a60189` | Inventario de `a0a5b982`: 1500 rutas, 1491 inventariadas, 8 hallazgos abiertos y 1 verificada. |

El recibo D2 conserva `integration_ref_final_cleanup` como reservado y
apunta a un recibo final entonces previsto; no acredita que esa limpieza
se ejecutara. Recuperar su archivo no ejecuta ni completa aquella acción.

`COMPOSICION_VALIDADA_OMEGA28_2026-10-08.json` es la copia exacta del
recibo original de `eb32ac96`, SHA256 raw
`7aeff888a2741da5bdc6ecdedac03164f380f48fe7e114accbd0a8a1d01da834`.
Describe las 334/0/1 ejecuciones históricas y conserva el estado
`Completed final validation; publication pending`. Su publicación actual
no convierte sus pruebas en ejecuciones de main actual ni exige editar
ese estado original. El recibo nuevo de cierre del integrador registrará
por separado el merge, el commit publicado, las verificaciones nuevas,
el OID remoto y la limpieza efectivamente realizada.

`RECUPERACION_HISTORICA_R4_2026-10-11.json` acredita únicamente siete
copias exactas, su normalización LF controlada y los backups; tampoco
certifica runtime. Respaldo externo: 253 artefactos, 27.957.712 bytes,
en `C:/Users/jhona/.codex/audit-evidence/trader-gemini/`
`r4-closure-2026-10-11-9cd26ca7-cfae-44ec-b72b-0258b301ba45`.
Los manifiestos conservados son:

- `manifest.json`, SHA256
  `22661d9cdb8443faaac0316446f27e2f03f98257444ed00671d31af8c8c7416b`.
- `extension-manifest-e7cf3772-b2e0-4365-a92c-e21c0b04cd46.json`, SHA256
  `899237e6e883b6153cd3d01d1b17672daa1b6edc543edc02aac62174b5a7c721`.

Los helpers anteriores permanecen sólo como evidencia archivada y no se
ejecutaron. La auditoría de esta adenda sólo leyó Git/recibos y actualizó
este informe; no escribió índice, refs, commits ni fuentes congeladas.

## Corte posterior de R4 — 13 de octubre de 2026

La validación vigente se ejecutó en `f62590e33a76a61a67320fed0e1ecc9ae0514ebd`: 713 pruebas aprobadas, cero fallos y siete ignoradas. T1 exacto pasó con 18/144 genes sensibles (12,5 %), cinta sintética, mínimo 0,110 intacto y un diagnóstico filtrado; no mide rentabilidad. Fuentes, genomas y candidatos de modelos permanecieron estables.

La composición con LH12 `92792116` quedó en `4191c65f`; se resolvieron dos conflictos documentales conservando historia y se aprobó all-targets en release antes del merge. La equivalencia posterior preservó los 664 inputs compilables y 637 Rust de T1. La publicación en main sigue pendiente en este corte local; su estado final se consulta en la [adenda del 13 de octubre](ADENDA_CIERRE_R4_2026-10-13.md).

Los recibos del 8 y 11 de octubre conservan sus fechas y alcances originales. Toda auditoría de archivo completo continúa pendiente; sólo los hunks de ceros/replay y sus contratos tienen evidencia mecánica acotada. Nulo condicional, familia efectiva, frescura/épocas y representación de cartera permanecen abiertos.