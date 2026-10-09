# Revisión estática de integración core / watcher / OU / registry

Fecha: 2026-10-08. Revisor: system_inventory. No Cargo, cambios de runtime, refs ni procesos.

Árbol candidato del índice: `2777578104f7931bb44f7e053103026b45ca811c`.
Padres observados: HEAD `f3f86960190d99aeacdb75b14cdfab0adceaa186`; MERGE_HEAD `18bbd1d90a073ae49ca8083c5898537013e4bfeb`.
Los siete blobs se revalidaron después en `7acf36aadf33a323b368f8e59fb77e2aafd40720`, árbol `f26f3b83e784dac821b53a5363f19bb29baf3c4d`: permanecen idénticos. Entre ambos árboles cambió sólo el test stateful_transition. La lectura inicial no halló diff unstaged en las rutas revisadas ni entradas unmerged.

## Dictamen de la composición

Aprobación estática acotada: no encontré pérdida bloqueante causada por esta integración. No equivale a compilar, ejecutar, certificar la teoría ni aprobar rentabilidad. El residual OU de abajo ya existe en el padre remoto y debe conservarse como pendiente antes de activar esa API en producción.

| Ruta | Blob del candidato | Evidencia |
|---|---|---|
| `crates/god-engine-core/src/lib.rs` | `b045efbf82766090478c9da17412b05acb7d2810` | Frente a remoto incorpora MG02/MG05 y módulos MW; frente a f3 conserva cambios de sombras/telemetría/nombres de remoto. |
| `crates/god-engine-core/src/stateful_engine.rs` | `55b3ed4820c2ea7a74737f0c411b9f5bae14aef4` | Conserva rename `last_exit_fastband_ms` y publica `last_event_ms` después de todos los guards de admisión. |
| `crates/god-engine-core/src/model_reload.rs` | `3357a6d870b5c7c6176958faa793324e0eec45f2` | Idéntico a f3: JSON/BIN seleccionados una vez, marca sólo tras éxito, errores reintentables. |
| `src/bin/god_engine.rs` | `eea831560cce74f00570b230724f5eef3f106420` | MW startup/polling conservados, Arc Darwin compartido, nueva tercera SL fallback con `sl_at_tau` preservada. |
| `crates/strategy-core/src/multivariate_coint.rs` | `3d87089163cf2011e7e9a4efb7d98c88985d4a51` | Idéntico a18bb; revisar residual OU, no inventar resolución. |
| `crates/strategy-core/src/vecm_arbitrage.rs` | `f3a2cc9d344995b7cdf40f7d96324fe2b42a3f4a` | Idéntico a18bb. |
| `crates/omniscient-registry/src/lib.rs` | `6cfd1a7b59d16ca6b88fea111a4667fdd06afbde` | Idéntico a18bb; claves en stack preservan formato exacto y fallback global/scoped. |

`evidence_publication.rs` también permanece idéntico a f3, blob `b0d34fe01fe4e08476ed05e8def2996fc4b3d30b`. Sus consumidores sobreviven en core: coherencia después de maduraciones (1667), dual directo (2749), feedback del cierre (3764), y Lundberg tras observar cierre (3750). La ruta espectral sigue consultando `coherencia_par_veto` / `significativo_familia`; no se debilitó Ville para madurar el fixture. La escala/refrescado y sentinel de retiro conservan la política revisada previamente.

Registry Ω25: buffers 96/64 bytes usan los mismos separadores/formato, sin truncar; claves largas conservan fallback a `format!`. La afirmación absoluta «sin heap» no cubre ese fallback ni creación de parámetros nuevos en `set`; no encontré cambio semántico de resolución de claves en la composición revisada. Latencia no medida por esta revisión.

## OU-R4-01 — Modo físico cae a proxy por evento

Estado: **CONFIRMADO_ESTATICO_NO_REPLAY**. Severidad propuesta MED para el contrato de la API opt-in. Alcanzabilidad en vivo no demostrada; no incidencia financiera atribuida.

En `multivariate_coint.rs:83–84` la documentación distingue modo físico y modo legacy. Sin embargo el bloque `physical_sde` de120–150 sólo termina la función cuando emite señal. Si está frío, no cruza el umbral, su vida media supera3600s o update devuelve0, continúa en el evaluador legacy desde153. No encontré un comentario/contrato que declare ese fallback; la cabecera conserva la advertencia de proxy legacy, pero no explica que ambos operan al activar SDE.

El fallback emite `SignalIntent` sin `expected_duration_ms` explícito (233–251), por lo que hereda0 del `Default` derivado en `types.rs:20–24`. Esto contradice el alcance anunciado de calibrar la intención física a la vida media; no prueba una orden de exchange ni pérdida de capital.

### Fixture mínimo propuesto, no ejecutado en Rust

1. `MultivariateCointegrationEngine::new([1.0,0.0,0.0,0.0],2.0).with_continuous_ou()`.
2. `update_and_evaluate(&[1.0,1.0,1.0,1.0],1000)` inicializa y devuelve None.
3. `update_and_evaluate(&[exp(0.1),1.0,1.0,1.0],1000)` (o500, retroceso).

SDE retorna z0 y sigue frío; legacy calcula count2, mean0.05, var0.000198, z≈3.553345, theta0.195 y half-life≈3.5546 periodos. Por lectura del control flow devuelve Short con `expected_duration_ms=0`. La aritmética fue contrastada en Python únicamente; no se presenta como ejecución del binario Rust.

## OU-R4-02 — Timestamp no creciente muta estado físico

También **CONFIRMADO_ESTATICO_NO_REPLAY**, mismo alcance opt-in. `vecm_arbitrage.rs:270–274` trata `count==0 || ts_ms<=last_ts_ms` en el mismo bloque: sobrescribe `last_value`, retrocede/repite `last_ts_ms`, incrementa count y devuelve0. Por tanto un timestamp igual/atrás NO se rechaza sin mutación. Puede fabricar madurez y dejar el próximo intervalo referido a una observación vieja. Ejemplo directo: primer update a1000, luego500; last_ts queda500 y count2.

La rama `dt_sec<1e-4` de278 tampoco expresa un rechazo tipado; con timestamps u64 en milisegundos, un incremento positivo es al menos1ms, así que ese umbral no intercepta duplicados/retrocesos (ya tomaron el bloque anterior).

## Alcanzabilidad y siguientes contratos

`rg` sobre `crates` y `src` localizó `with_continuous_ou` sólo en su definición y el test unitario `test_multivariate_cointegration_continuous_ou_physical_clock`; constructores de `MultivariateCointegrationEngine` sólo en sus tests y `basket_state_contract`, además de reexport público en strategy-core/lib.rs. No caller productivo encontrado. No descarta consumidores externos de esta API pública.

Antes de cablearla a producción, acordar si el modo SDE es exclusivo o un híbrido explícito; probar cold/startup, dup/backward, vida media larga, z subumbral y tau de toda señal. No corregir silenciosamente dentro de este merge ni mezclar los tests legacy con prueba de calibración física. El coordinador y el integrador recibieron estos hallazgos; no se pidió activar runtime.

## Adenda: reproducción Rust aislada solicitada por el coordinador

El estado de OU-R4-01/02 se actualiza a **REPRODUCIDO_RUST_AISLADO_NO_WORKSPACE_NO_LIVE**. Se conservan arriba el razonamiento y alcance estáticos originales.

Directorio de evidencia: `ou_fixture_2026-10-08_7ef8e782/` junto a este informe. `receipt.json` contiene commit, blobs, rangos, SHA256, argv exactos, exits y duraciones; los archivos01.log/02.log registran compilación y ejecución. No se modificaron las fuentes extraídas.

- Fuente fija: `18bbd1d90a073ae49ca8083c5898537013e4bfeb`.
- `types.rs` completo, líneas1–43, SHA256 `407f3298fbfffcce11d698db652fe7387b0587c940b86b05be04a0ecd8b6e766`.
- `multivariate_coint.rs` completo, líneas1–526, SHA256 `4870ed67d5473acf032e1d27d743ea9bda0ce842b176637aca517f5990565138`.
- Bloque SDE exacto de `vecm_arbitrage.rs`, líneas197–337, SHA256 del fragmento `dabb8d2a2c86df2ec57fd7b70420893399538056924c953b5e892393619e49e3`.
- Compilador invocado mediante `rustup run nightly-2026-06-30 rustc --edition=2021 --test runner.rs -o ou_diagnostic.exe`; exit0,1.224s. No Cargo ni enlace de workspace.
- Ejecutable filtrado a `r4_ou_diagnostic:: --nocapture --test-threads=1`: **2 passed,0 failed,8 filtered out**, exit0,0.114s de proceso. Los tests comprueban la PRESENCIA del defecto; el verde no significa que se haya reparado.
- Ambas secuencias1000→500 y1000→1000 produjeron `physical_z=0`, `sde_count=2`, `legacy_count=2`, `legacy_z=3.553345272594`, `signal=Short`, `duration_ms=0`; el primer caso además deja `sde_last_ts=500`.
- SHA256 del log de ejecución02.log: `671c42e788125879e23a6308abbd589f29cab8e8069646b385d164b962ff4e16`.

Esta evidencia ejecuta los métodos Rust exactos del padre remoto en un harness std-only con dos eventos sintéticos. No ejecuta dependencias completas, binarios del proyecto, órdenes, replay de mercado ni flujo vivo. La búsqueda de alcanzabilidad y el límite de incidencia financiera permanecen iguales.

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
