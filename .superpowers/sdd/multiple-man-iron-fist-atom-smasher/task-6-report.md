# Task 6 — Raw archive: recuperación, durabilidad y entorno

## Estado: BLOCKED

La puerta Clippy de líneas cambiadas encontró un diagnóstico propio. Siguiendo el brief, se conservó la primera causa y se detuvo el trabajo sin repetir la puerta para buscar verde. No se creó commit ni `refs/agent/v35/f4/raw-archive-recovery`.

- Base exacta: `5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9`.
- Archive aislado: `.agent_tx/staging/KIMI-QUANT-SR-V35/task-6-raw-archive-recovery`.
- Commit: ninguno.
- Ref privada: no creada.
- `main` permanece en `814598b32cdcb25512ad829a33734f11e4b12d19`.
- `origin/main` permanece en `031a8ddceb6bc974a44f697ab13fbdb9cb78eceb`.

## Cambios implementados, pendientes y sin commit

- `crates/storage-engine/src/raw_zone.rs`: recupera ticks completos sin footer y reconstruye count/FNV; publica un activo con footer válido; rechaza sufijos parciales sin modificar el archivo; añade nombres finales con timestamp/PID/contador; escribe footer, flush y `sync_all`; restaura el writer en fallos anteriores al rename; abre un activo nuevo tras publicar; sincroniza directorio en Unix; propaga errores de `SystemTime`; `flush` cerrado devuelve `NotConnected`.
- `crates/data-pipeline/src/ws_client.rs`: resuelve el archivo mediante `quantum_arena::paths::env_data_path("raw_ticks")`; networking y demás rutas sin cambios.
- Nuevos contratos `raw_zone_recovery_contract.rs` y `raw_archive_environment_contract.rs`.
- Delta rastreado antes del bloqueo: exactamente esos cuatro archivos Rust, 290 inserciones y 65 eliminaciones según `git diff --stat` (incluye los dos archivos nuevos).

## Entorno de ejecución

- `CARGO_TARGET_DIR=C:\Users\jhona\Documents\Proyectos\Trader Gemini\target` (canónico).
- `CARGO_BUILD_JOBS=16`.
- `RUSTC_WRAPPER=sccache`.
- `RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C linker=rust-lld`.
- Toolchain: `nightly-2026-06-30`.
- Sin semáforos, targets alternos, red, operaciones financieras, procesos de trading ni credenciales.

## RED auténtico, antes de modificar producción

### Storage

Comando:

```text
cargo +nightly-2026-06-30 test -p storage-engine --locked --test raw_zone_recovery_contract -- --test-threads=1
```

Código de salida: **101**. Resultado: **0 passed / 4 failed / 0 ignored / 0 filtered**.

Fallos esperados y observados:

- `open_recovers_complete_ticks_without_footer_and_appends`: sólo sobrevivió el tercer tick; faltaron los dos originales.
- `open_publishes_active_with_valid_footer_before_new_appends`: se obtuvo 1 segmento en vez de 2.
- `open_rejects_partial_suffix_without_touching_active_file`: `open` aceptó el sufijo parcial.
- `repeated_writers_in_one_second_never_replace_published_segments`: quedó 1 segmento en vez de 12 por colisiones de nombre.

### Entorno

Comando:

```text
cargo +nightly-2026-06-30 test -p data-pipeline --locked --test raw_archive_environment_contract -- --test-threads=1
```

Código de salida: **101**. Resultado: **1 passed / 1 failed / 0 ignored / 0 filtered**.

- Pasó `raw_ticks_has_separate_demo_and_production_paths`.
- Falló `ws_client_delegates_raw_archive_path_to_environment_policy`: `ws_client` no consumía `env_data_path("raw_ticks")`.

## GREEN focal posterior a la implementación

- Storage, mismo comando RED: exit **0**, **4 passed / 0 failed / 0 ignored / 0 filtered**.
- Entorno, mismo comando RED: exit **0**, **2 passed / 0 failed / 0 ignored / 0 filtered**.

## Validaciones completas

| Comando | Código | Resultado |
|---|---:|---|
| `cargo +nightly-2026-06-30 test -p storage-engine --locked -- --test-threads=1` | 0 | **47 passed / 0 failed / 0 ignored / 0 filtered**; doc-tests 0/0. Desglose: 17 lib + 7 mmap admission + 1 mmap diagnostics + 11 ownership + 7 corrupción + 4 recuperación. |
| `cargo +nightly-2026-06-30 test -p data-pipeline --locked --test raw_archive_environment_contract -- --test-threads=1` | 0 | **2 passed / 0 failed / 0 ignored / 0 filtered**. |
| `cargo +nightly-2026-06-30 check --workspace --all-targets --locked` | 0 | `Finished` en 6m44s. Warnings baseline en `os-guardian` y `god_engine`; ninguno atribuido al delta por la puerta. |
| `rustfmt +nightly-2026-06-30 --edition 2021 --check crates/storage-engine/src/raw_zone.rs crates/storage-engine/tests/raw_zone_recovery_contract.rs crates/data-pipeline/src/ws_client.rs crates/data-pipeline/tests/raw_archive_environment_contract.rs` | 0 | Cuatro archivos con formato limpio. |
| `git diff --check 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9 -- <cuatro archivos Rust>` | 0 | Sin errores de whitespace. |
| `python -I -B scripts/puerta_lineas_cambiadas.py --base 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9 --solo clippy` | **1** | **BLOCKED**: 1 diagnóstico en líneas cambiadas; 0 errores de build. |

## Primera causa conservada

La salida exacta de la puerta fue:

```text
Líneas .rs añadidas o modificadas frente a 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9: 290 en 4 archivos
clippy: deuda total 793 diagnósticos en 257 archivos; en líneas cambiadas 1; errores de build 0
::error file=crates/storage-engine/src/raw_zone.rs,line=151,endLine=151,title=clippy::warning[clippy::chunks_exact_to_as_chunks]: using `chunks_exact` with a constant chunk size
1 hallazgos en líneas cambiadas o build roto.
```

No se editó esa línea después del fallo, no se repitió la puerta y no se creó el commit/ref condicionado a que todo pasara.


---

## Fix round 1 — corrección y cierre

Se cambió únicamente el nuevo bucle de recuperación de `crates/storage-engine/src/raw_zone.rs`:

```rust
for (index, chunk) in bytes.as_chunks::<40>().0.iter().enumerate() {
```

El resto del archivo, incluido `verify_segment`, quedó intacto. La sección BLOCKED anterior conserva el historial de la primera ejecución; esta ronda repite la puerta y todas las validaciones necesarias sobre el sujeto corregido.

### Puerta repetida con Git real e índice privado

Entorno Git usado explícitamente:

```text
GIT_DIR=<raíz>/.git
GIT_WORK_TREE=<archive>
GIT_INDEX_FILE=<archive>/.task6-git/index
```

Comando:

```text
python -I -B scripts/puerta_lineas_cambiadas.py --base 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9 --solo clippy
```

Código de salida: **0**. Resultado:

```text
Líneas .rs añadidas o modificadas frente a 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9: 290 en 4 archivos
clippy: deuda total 792 diagnósticos en 257 archivos; en líneas cambiadas 0; errores de build 0
Limpio: las líneas cambiadas pasan clippy.
```

### Revalidación posterior a la corrección

| Comando | Código | Resultado |
|---|---:|---|
| `cargo +nightly-2026-06-30 test -p storage-engine --locked -- --test-threads=1` | 0 | **47 passed / 0 failed / 0 ignored / 0 filtered**; doc-tests 0/0. |
| `cargo +nightly-2026-06-30 test -p data-pipeline --locked --test raw_archive_environment_contract -- --test-threads=1` | 0 | **2 passed / 0 failed / 0 ignored / 0 filtered**. |
| `cargo +nightly-2026-06-30 check --workspace --all-targets --locked` | 0 | `Finished` en 1m50s. |
| `rustfmt +nightly-2026-06-30 --edition 2021 --check crates/storage-engine/src/raw_zone.rs crates/storage-engine/tests/raw_zone_recovery_contract.rs crates/data-pipeline/src/ws_client.rs crates/data-pipeline/tests/raw_archive_environment_contract.rs` | 0 | Cuatro archivos con formato limpio. |
| `git diff --check 5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9 -- <cuatro archivos Rust>` | 0 | Sin errores de whitespace. |

Todas las validaciones requeridas quedan GREEN en esta ronda. El commit atómico se crea con padre `5e4a1d1f78bd9720b6e52e9b18821ebd1e4ecbd9` y se registra únicamente en `refs/agent/v35/f4/raw-archive-recovery`; su SHA se obtiene con `git rev-parse refs/agent/v35/f4/raw-archive-recovery`. No se actualiza `main`, `origin/main` ni ningún remoto.
