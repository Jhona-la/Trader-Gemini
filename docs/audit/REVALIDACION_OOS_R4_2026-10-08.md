# R4 — Adenda OOS y medición económica, 2026-10-08

Revisión estática de las 14 rutas de `PROTOCOLO_OOS_R4_2026-10-07.md`. Fuentes fijadas: base `639c3e0d7c5f0dc8c376ac04af2a2fdc7a5bc5a9`, remoto `18bbd1d90a073ae49ca8083c5898537013e4bfeb`, árbol candidato `f26f3b83e784dac821b53a5363f19bb29baf3c4d`. El árbol quedó fijado después por el commit local `7acf36aadf33a323b368f8e59fb77e2aafd40720`; no es un recibo de publicación. No se modificó el informe histórico, no se inspeccionaron filas de datos y no se ejecutó Cargo/T1/OOS.

## Redacción corregida aplicable al protocolo

**Causalidad de estimación:** «Para replay histórico debe acreditarse que la estimación usa exclusivamente información disponible antes de cada decisión evaluada, incluyendo disponibilidad y maduración de etiquetas; la fecha material de creación del archivo no sustituye esta trazabilidad. Un examen prospectivo exige además fijar previamente el protocolo y registrar todas las consultas».

**Ejecución y reproducibilidad:** «El examen se ejecuta según el protocolo preregistrado, conservando salidas completas, fallos, ventanas sin operar y desviaciones. Las réplicas y semillas deben estar previstas; una reejecución idéntica sólo comprueba reproducibilidad y no añade evidencia independiente. Retocar decisiones usando sus resultados consume ese examen como datos de selección y exige una nueva reserva para el examen final».

En la fila R6, usar «reglas de adaptación e información permitida preregistradas, estado inicial identificado y genealogía de cada actualización». Esto permite aprendizaje causal prescrito y coincide con la aclaración adaptativa del borrador; no permite entrenamiento externo sin registrar ni acceso a etiquetas futuras.

## Delta económico verificable

- `booktick_replay.rs`: el único delta vs639 son tres líneas de comentario sobre fallback de SL. Permanecen final/DD de capital realizado, Sharpe por trade, símbolo único y span del tape con warm-up. No aparece un nuevo patrimonio MTM en este consumidor.
- `continuous_evolution_backtest.rs`: sí hay una mejora material de medición que debe reconocerse. `daily_report` compara patrimonio final con patrimonio anterior, acumula diferencias telescópicas y publica residual de reconciliación. El doble conteo del flotante arrastrado tiene corrección fuente y dos tests nuevos presentes; esta revisión no ejecutó esos tests. El patrimonio terminal conserva hipótesis de comisión de salida0.0005, fallback de precio y omisión de flotantes no finitos. Es un reporte diario del harness, no una curva sincronizada certificada del core vivo ni validación de liquidación/funding/costes. También cambian bounds de nichos y sanitización; no transferir resultados históricos al candidato sin nuevo recibo.
- `selection_stats.rs`: cambia floor de curtosis/fallback skew. El testigo Rust exacto actual y las limitaciones se documentan en `REVALIDACION_FUNDAMENTOS_R4_2026-10-08.md`; no llamar HAC a esta fórmula. Dependencia, contaminación y dispersión entre ensayos siguen sin contrato suficiente para interpretar DSR nominal como error global.
- `src/bin/evolution.rs`: el candidato conserva recuperación OOS/SA que origin18bb no incluía. Valida segmentos antes de modelos y usa prefijo IS disponible; usa selección/penalización explícitas y deja de etiquetar utilidad interna como retorno de tres días. No incorpora una curva MTM ni un ledger de examen sellado. El desacople de mutaciones TP/SL legacy y curvas continúa aparte.
- El plan por archivos cambia como documentación. `metrics`, esquema `ml_registry`, `train_forest`, manifests y cuatro ADR conservan sus blobs de639 dentro de estas14rutas; no se han revalidado datasets/modelos cargados en vivo por esa igualdad.

Continúan faltando recibos concretos de: cobertura temporal y as-of por activo/label; patrimonio común y flujos externos reconciliados; costes sin doble conteo; hashes efectivos de modelos y configuración, flags/roster y genealogía; reloj/fronteras de72h, tratamiento de solape/dependencia y protocolo de consultas/custodia. Las ventanas disjuntas evitan solape mecánico pero no garantizan independencia. La medición diaria corregida es progreso parcial y no demuestra por sí sola la meta de100% cada72h. T1/paridad preservan contratos y no certifican retorno económico.

## Comparación de blobs

Las siguientes rutas cambian entre639 y el candidato; la columna18bb distingue evolución remota de recuperaciones locales. Una diferencia entre padres no acredita una regresión temporal.

| Ruta | base639 | origin18bb | candidato f26 |
|---|---|---|---|
| `docs/PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md` | `4d2902323ae179ab64d6adf061a1db44379c3315` | `1cdb604bc6686e959b00e3836e824d0b8589bd2d` | `1cdb604bc6686e959b00e3836e824d0b8589bd2d` |
| `crates/backtest-engine/src/booktick_replay.rs` | `1ed78cc9673f1300b2c44a418360f53e9e5d86f5` | `7237beca53dcd5664a7f90bab01ae521e2bffdeb` | `7237beca53dcd5664a7f90bab01ae521e2bffdeb` |
| `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs` | `114ab1a47db5d8c16796962a845f16f0d3d29730` | `a8abdeded7bc1ab4e8d32d697bdfe336f8afc05d` | `a8abdeded7bc1ab4e8d32d697bdfe336f8afc05d` |
| `crates/risk-engine/src/selection_stats.rs` | `2bb2fa0d0a413a3d386852341318497959324298` | `1e2ba9b194cf133bc614071c168003287558145d` | `1e2ba9b194cf133bc614071c168003287558145d` |
| `src/bin/evolution.rs` | `44d7d0f58a4cea8efcaf1cc788ad69094a47c7b9` | `44d7d0f58a4cea8efcaf1cc788ad69094a47c7b9` | `30f34f1b205edf7929fbcfa8ace181d5a546e1d4` |

Cambian 5 de14rutas; permanecen 9 idénticas a639 en el candidato:

- `crates/backtest-engine/src/metrics.rs`.
- `crates/god-engine-core/src/ml_registry.rs`.
- `src/bin/train_forest.rs`.
- `config_dir/models_manifest.json`.
- `config_dir/copulas_manifest.json`.
- `docs/adr/ADR-0003-doctrina-meta-geometrica.md`.
- `docs/adr/ADR-0008-reloj-revalidacion-modelos.md`.
- `docs/adr/ADR-0009-regeneracion-mensual-copulas.md`.
- `docs/adr/ADR-0010-arquitectura-dl-modular.md`.

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
