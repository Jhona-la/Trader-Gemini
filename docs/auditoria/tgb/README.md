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

## Fallos preexistentes de la compuerta del cierre (k40 y k41)

Cambio 2 de la N1 de AGY-CLI sobre `c0f3bc1f7` (`agentes\salidas\20261117-2000-agy-n1tgbg\informe.md`, PUNTO 3 y
PUNTO 5). En la compuerta del cierre, las dos suites enteras repetidas sobre `c0f3bc1f7` dieron:

- `k40_gec_qa_repeticion` (`test --locked -p god-engine-core -p quantum-arena --no-fail-fast`): 1415 pasan, 6 fallan,
  49 ignoradas. sha256 del log `E60376DA732E77E2D16A173DA9F755BEE8545AED86244A886BFFC7D2DA48195D`.
- `k41_bt_repeticion` (`test --locked -p backtest-engine --no-fail-fast`): 200 pasan, 1 falla, 20 ignoradas.
  sha256 del log `1BE46BFD76F4B4202725C0F03164092B4CDB2E2268CAC1D21497B00B1F8F5F85`.

Los logs estan en `integrador\tgbg_logs` (fuera del repo; `k41` pesa 22,6 MB). Los 7 fallos ya existian antes de TGB-G.
Esta tabla los cruza con la evidencia anterior al primer commit TGBG-1 (`40616b7fd`, 17 nov 11:03):

| Fallo (k40/k41) | Ya fallaba en | Evidencia anterior a TGB-G |
| --- | --- | --- |
| `test_c2_wave27_connes_kreimer_divergence_veto_propagation`, `test_c3_wave27_donaldson_thomas_wall_crossing_veto_propagation`, `test_c4_wave27_asymptotic_safety_singularity_veto_propagation` (god-engine-core, `f99_002_wave27_living_graph_synthesis_contract`) | rotos por `55a3fd464` (W29, 10 nov 14:21) | `20261112-2314-qoder-PRUEBAS-ROJAS.md:5`; certificados de `50fc038e1` (f07) y `e62ace4ca` |
| `glm223_warmup_nan_y_publicacion_en_frontera`, `glm223_colas_efectivas_aplanan_la_geometria` (god-engine-core, `ricci_q_publicacion_contract`) | desde su nacimiento, `fb072846e` (GLM-223, 10 nov 13:02) | `20261112-2314-qoder-PRUEBAS-ROJAS.md:7`; `20261114-2233-glm-comprobacion-ricci-q-glm223.md:3` (colision de escala: sigma de 1e-3 contra h absoluto de 1e-3); certificados de `50fc038e1` (f07) y `e62ace4ca` |
| `rb_p1_ningun_paquete_sube_su_cuenta_de_panicos` (quantum-arena, `rb_p1_trinquete_panicos_contract`; telemetry-server 12 > linea base 4) | origin/main `50fc038e1` | certificado de `50fc038e1` (f07, "rb_p1 telemetry-server 12>4"); `20261116-0307-integrador-PUBLICACION-C3.md:6` (CI de `5ef982bb7` con los mismos 7 fallos registrados); certificado de `e62ace4ca` ("RB-P1 mismas filas") |
| `t1_cobertura_genetica_del_oraculo_de_aptitud` (backtest-engine; 9,7 % < 11 %) | origin/main `50fc038e1` | `20261112-2314-qoder-PRUEBAS-ROJAS.md:9`; `20261113-1518-qoder-V4K-MAPA.md:7`; certificados de `50fc038e1` (f08, 194/1/20) y `324ef18db` (q08, 200/1/20) |

Comparacion por cuentas con la base: `e62ace4ca` dio 1415/6/49 en god-engine-core + quantum-arena, con los mismos 6
por nombre. `324ef18db` dio 200/1/20 en backtest-engine, con el mismo fallo. k40 y k41 repiten esas cuentas exactas: TGB-G
no agrega ni quita fallos por nombre.

`.agent_tx/coordinacion/REGISTRO_DE_FALLAS.md` (68 lineas al 18 nov) se consulto por pedido de la N1 y no tiene una fila
FA para estos 7 fallos. La preexistencia se apoya en los mensajes y certificados de arriba, no en ese registro. Abrir una
fila es decision del coordinador.

Los documentos citados no estan en el repo (`.agent_tx/` y `integrador\certificados\` estan fuera de git). Los hashes fijan
los bytes leidos el 18 nov:

| Documento | sha256 |
| --- | --- |
| `.agent_tx/coordinacion/mensajes/20261112-2314-qoder-PRUEBAS-ROJAS.md` | `463A4E3BD63D2FC727EBCD4F0923E3E0753829B99DF0A39EEE4510CD42B9BCCD` |
| `.agent_tx/coordinacion/mensajes/20261113-1518-qoder-V4K-MAPA.md` | `A9D4BBD82BBADAD8E616408650ADE3DF7542636E3810F76865BB0DBB33B312DF` |
| `.agent_tx/coordinacion/mensajes/20261114-2233-glm-comprobacion-ricci-q-glm223.md` | `A050E77015F836D27FC039849953F0436F3F3DABCB9C5121F7C720F19D99F7A0` |
| `.agent_tx/coordinacion/mensajes/20261116-0307-integrador-PUBLICACION-C3.md` | `0B8CE26E8197B9E9D246D627054B044498621F1D6556510298D956A2116AFAB6` |
| `.agent_tx/coordinacion/REGISTRO_DE_FALLAS.md` | `1A51502B10A1297A4407FD0B6D550E88BF9573661E23F4D5CC935C56B4E614D1` |
| `integrador\certificados\50fc038e1ebf3f3ac3f94628a51de99e9ff38f72.json` | `CBBE203FC990C0581C623318EC7EB4BD519D106CE1DAA5E0E151BCDAFBB602CF` |
| `integrador\certificados\e62ace4cad16389e5586e82e959620eddd8e26d0.json` | `BEB1348EA46A28ABA5950897581BCA0A481969703000338C0C6E1E7305304CD2` |
| `integrador\certificados\324ef18dbe1805320db70056df4fe5d6a7edd329.json` | `27FB9E8290C95A4B83005C8362C85E9AF6D73156475AF219C583E4894960A82F` |
| `integrador\certificados\c0f3bc1f7c45878f6c0fdaa98090ad5b520744b4.json` | `A7DD5305A9D16D2363220F95425E10BCF68CF709009AADB6C882060BC437A6A5` |

## Ampliacion de la lista blanca aprobada (cambio 1 de la N1)

El coordinador aprobo el 17 nov a las 20:45 que el Lote 17 (TGB-G-PROCEDENCIA) incluya `docs/auditoria/tgb/.gitattributes`
(`* -text -whitespace`, para conservar los bytes de los logs y parches) y los guiones auxiliares de
`docs/auditoria/tgb/runs/herramientas_*/`. Esos archivos no estaban en la lista cerrada de `pedido_tgb_g_v1.md` §5.2. La
ampliacion queda anotada en la adenda de la lista blanca del Carril E
(`.agent_tx/diagnostics/auditoria_perdidas/agy_cli_lista_blanca_carril_e_v2.md`).
