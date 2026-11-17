# Registro de evidencia de la serie TGB (lector .tgb y cierre TGB-G)

Pedido: `pedido_tgb_g_v1.md` (CODEX-CLI, 20261117-0015), seccion 2. Rama `integrador/capital-v4l`.

## Estructura

```text
docs/auditoria/tgb/indice.jsonl                       una linea por corrida (paso, run_id, tipo, estado, resultado, base, ruta, sha256 de checksums)
docs/auditoria/tgb/runs/<paso>/<sha_base>/<run_id>/
    stdout.txt        salida original del lanzador, bytes intactos (stdout y stderr combinados: cmd 2>&1)
    run.json          tipo, argv, filtro, esperado/observado, codigo de salida, duracion, cuentas de pruebas, asercion causante
    subject.json      commit base y su tree, blobs de Cargo.lock y .cargo/config.toml, herramientas, perfil, CARGO_TARGET_DIR
    subject.patch     cambio evaluado sobre la base (git diff HEAD + archivos nuevos), tomado al lanzar la corrida
    mutation.patch    reemplazo exacto de la mutacion (archivo, sha256 previo, viejo/nuevo), solo en mutaciones
    checksums.sha256  sha256 de cada archivo de la carpeta
docs/auditoria/tgb/runs/herramientas_*/              guiones de lanzamiento y de mutacion (.txt)
docs/auditoria/tgb/procedencia/                       formatos del recibo, la evidencia y el indice revisado
```

## Como comprobar y reconstruir un sujeto

1. `cd runs/<paso>/<sha_base>/<run_id> && sha256sum -c checksums.sha256`, y comparar el sha256 de `checksums.sha256` con su linea en `indice.jsonl`.
2. Reconstruir: `git worktree add <tmp> <sha_base>`, `git -C <tmp> apply subject.patch`. El arbol resultante es el sujeto de la corrida.
3. Repetir con el `argv` de `run.json` por `scripts/cargo_semaforo.sh` sobre el target compartido.

`tested_commit` (la base del sujeto) y el commit que publica este registro (`evidence_commit`) son distintos: un verde de
aqui no se traslada al commit de documentacion ni a otra punta. El certificado del SHA final va aparte
(`integrador\certificados\<sha>.json`).

## Estados

- `atribuible`: corrida del cierre TGB-G (TGBG-1..5) con su parche de sujeto.
- `historico_cabecera_limpia`: compuerta de 324ef18db (q*, m_r00): cabecera con 0 archivos rastreados modificados; no registra los no rastreados.
- `historico_no_atribuible`: serie A-G de lector-tgb (`integrador\tgb_logs`). Cada log nombra su HEAD, pero se tomo con
  cambios sin confirmar cuyo contenido no se guardo: no acredita su sujeto. Se conservan sin reescribir; no son una
  repeticion ni reconstruyen el pasado. Los reemplazos de sus mutaciones estan en `runs/herramientas_lector_tgb/`.
- `excluido_por_tamano`: `q08_bt` (22,6 MB), solo en el indice con su sha256 y su ubicacion.

Controles CIEGOS o no validos, visibles y sin contar como proteccion: `g_m3`/`g_m3b` (ventana de 0 ulp de H, ciega),
`cd_m4` (no compilo; repetido como `cd_m4b`). En el cierre, `p5_m02` es un control de ANCLA de texto. Las corridas
reemplazadas (rojos repetidos, verdes previos a un refuerzo) llevan `nota` en su `run.json`.

## Matriz G01-G12 (cierre TGB-G)

| ID | Prueba | Clase | Rojo antes (sujeto) | Verde despues | Mutaciones |
| --- | --- | --- | --- | --- | --- |
| G01 | `tick-tape --test tgb_g_contract` `tgb_g01_recetas` | PURA | regresion (pasa con la logica de H) | p1_r02, p2_r02 | gg_sin_exclusion ROJA |
| G02 | `tgb_g02_cruce_y_vecinos` | PURA | regresion reforzada | p1_r02, p2_r02 | gg_medio_solo ROJA (binade) |
| G03 | `tgb_g03_rechaza_autodeclaracion` + lib `tgb_g_03_puente_*` | PURA | p2_r00 ROJO | p2_r02, p3_r04 | gg_productor ROJA |
| G04 | `tgb_g04_cadena_rota` + lib `tgb_g_04_puente_*` | PURA | p2_r00 ROJO | p2_r02, p3_r04 | gg_productor, gg_evidencia_hash, gg_indice_pin ROJAS |
| G05 | `tgb_g05_identidad_original` | PURA | regresion | p2_r02 | gg_hash, gg_bytes ROJAS |
| G06 | `tgb_g06_cobertura_y_cola` | PURA | p2_r00 ROJO | p2_r02 | gg_cobertura, gg_prefijo, gg_firma_discordante ROJAS |
| G07 | `tgb_g07_mixta_invalida_y_medio_tick` | PURA | p1_r01 ROJO (cantidad NaN admitida) | p1_r02, p2_r02 | gg_sin_exclusion, gg_prefijo ROJAS |
| G08 | `tgb_g08_cabecera_y_recibo` | PURA | p2_r00 ROJO | p2_r02 | gg_cabecera ROJA |
| G09 | `--bin dataset_admission` `tgb_g_09_dataset_exige_procedencia` | PURA | p4_r02 ROJO | p4_r03 | gg_dataset ROJA |
| G10 | `--bin train_forest` `tgb_g_10_todas_las_fuentes_antes_de_muestras` | PURA | p5_r00 ROJO | p5_r01 | gg_fuentes_diferidas ROJA; gg_main_sin_compuerta (ancla) ROJA |
| G11 | `--lib` `tgb_g_11_snapshot_y_evidencia_vigentes` | PURA | p3_r03 ROJO (= gg_snapshot) | p3_r04 | gg_snapshot ROJA |
| G12 | `tgb_g12_bin_tgb_equivalentes` + lib `tgb_g_12_puente_*` | PURA | regresion | p2_r02, p3_r04 | gg_hash_transporte ROJA |

Los rojos se tomaron sobre la logica de H trasladada a la costura nueva (la costura de cada paso esta en el
`subject.patch` de su rojo). Las funciones nuevas ausentes de H no se usaron como rojo por compilacion.
