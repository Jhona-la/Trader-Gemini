# Datos de la Línea H (Claude, LH-n) — 2026-10-12

Instrumento: `crates/backtest-engine/tests/t3_ventaja_conocida.rs` (T-3).
Plan y lectura de los resultados: `docs/PLAN_LINEA_H_CLAUDE_2026-10-12.md` §6.

| Archivo | Qué es | Cómo se regenera |
|---|---|---|
| `t3_escalera_13usd.tsv` | La escalera completa (8 peldaños) con 13 USD; una fila por peldaño, cabecera con el nombre de cada columna | `cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture --test-threads=1 t3_escalera_de_ventaja_conocida` y filtrar `[T3-TSV]` |
| `t3_lineas_13usd.txt` | Las mismas filas en la forma legible `[T3] …` | ídem, filtrar `[T3]` |
| `t3_trazas_13usd.txt` | Contadores del runner nativo por peldaño (`opened`, `council_vetoes`, `ml_vetoes`, `closes`) | ídem, filtrar `[T1-DIAG]` (el test fija `TG_TRACE_NATIVO=1`) |
| `t3_escalera_capital.tsv`, `t3_trazas_capital.txt` | Los dos peldaños fuertes con 100 y 1 000 USD | `… t3_escalera_por_capital` (`T3_CAPITALES=100,1000`) |
| `t3_capital100_max_fee_035.tsv`, `t3_trazas_capital100_max_fee_035.txt` | Intervención: 100 USD con el gen `max_fee_pct` en 3,5 % | `T3_CAPITALES=100 T3_MAX_FEE_PCT=0.035 … t3_escalera_por_capital` |
| `t3_lado_vs_estado.txt`, `lado_vs_estado.py` | Fracción de aperturas trazadas del lado del estado oculto, τ mediana y rama por peldaño (sólo las 100 primeras aperturas que el núcleo traza) | `python -I lado_vs_estado.py <log de la corrida>`; el script replica la LCG del generador y es sólo diagnóstico (base `e0cd88b7`) |
| `lectores.tsv`, `predicciones.tsv`, `lectores_resumen.md`, `lectores_raw.json` | Los tres lectores de la cadena meta → orden (H1 tamaño, H2 evidencia/EV, H3 dirección/τ): hallazgos con cita, predicciones falsables y resúmenes | workflow `linea-h-lectores` (tres `Plan` agents de sólo lectura) |
| `verificacion.tsv`, `escepticos_raw.json` | Veredicto de un escéptico por lente sobre cada hallazgo y cada predicción, con la escalera medida en la mano | workflow `linea-h-escepticos` |
| `ola2_adaptacion.tsv`, `ola2_adaptacion_lineas.txt` | Ola 2: E1 (ventaja larga, cuartiles por índice de cierre), E2 ventaja → nulo, E3 nulo → ventaja, E4 ventaja → antipersistencia; «segundo tramo» = cierres posteriores al último del primer tramo en frío | `… t3_adaptacion_en_el_tiempo` |
| `ola2_misma_serie_dos_veces.tsv`, `ola2_misma_serie_dos_veces_lineas.txt` | E1b: la misma serie dos veces (sólo cambia el estado aprendido) en tres regímenes | `… t3_adaptacion_misma_serie_dos_veces` |
| `ola2_lectores.tsv`, `ola2_lazos.tsv`, `ola2_predicciones.tsv`, `ola2_lectores_resumen.md`, `ola2_lectores_raw.json` | Ola 2: tres lectores de los lazos que aprenden por cierre (46 lazos con etiqueta, memoria, consumidor y escala), hallazgos y predicciones de E1–E4 | workflow `linea-h-lectores-adaptacion` |
| `ola2_verificacion.tsv`, `ola2_predicciones_veredicto.tsv`, `ola2_nuevos_escepticos.tsv`, `ola2_escepticos_sintesis.md`, `ola2_escepticos_raw.json`, `ola2_correcciones.md` | Ola 2: veredictos de los escépticos con la medición, predicciones contrastadas, hallazgos nuevos y correcciones al inventario de los lectores | workflow `linea-h-escepticos-ola2` |
| `ola2_post_pr62_escalera_13usd.tsv`, `ola2_post_pr62_valle_de_capital.tsv`, `ola2_post_pr62_adaptacion.tsv`, `ola2_post_pr62_adaptacion_lineas.txt` | Re-medición sobre el árbol con el PR #62 de la Línea C (`fc2ba2ec`): idéntica salvo E4 (`estrategia_falsada` = 31) | los mismos comandos sobre ese árbol |

| `ola3_escalera_v2_13usd.tsv`, `ola3_escalera_v1_vs_v2.tsv` | Ola 3: la escalera de 13 USD con el generador v2 (ruido gaussiano, mechas y volumen independientes) y el filtro de Bayes como referencia de cierres; y la comparación peldaño a peldaño con la v1 | `T3_GENERADOR=v2 … t3_escalera_de_ventaja_conocida` |
| `ola3_valle_v2.tsv` | Ola 3: el valle de capital con la v2 | `T3_GENERADOR=v2 … t3_valle_de_capital` |
| `ola3_adaptacion_v2.tsv`, `ola3_adaptacion_v2_lineas.txt` | Ola 3: E1–E4 con la v2, más E4' (anti → anti) y E5 (antipersistencia larga por cuartiles) y el pago del segundo tramo y del fresco en la fila | `T3_GENERADOR=v2 … t3_adaptacion_en_el_tiempo` |
| `ola3_escepticos.tsv`, `ola3_confirmaciones.tsv`, `ola3_escepticos_sintesis.md`, `ola3_escepticos_raw.json` | Ola 3: dos escépticos (matemática del filtro; generador y reproducibilidad) sobre el T-3 v2 | workflow `linea-h-escepticos-ola3` |

Generadores: V1 conserva la serie histórica y su fuga conjunta OHLCV. V2
separa los sorteos y usa ruido gaussiano; el canal intravela H1-01(iii) del
runner permanece. El filtro HMM de cierres no es un techo universal del motor.
La serie V1 se preserva; sus tablas dependen además del motor y del esquema.

Los archivos `ola3_*` recuperados son históricos: no hay huella verificable
del binario exacto que los generó, y sus columnas de referencia anteceden las
correcciones del 15Oct. No regenerarlos silenciosamente con el código actual.
`ola3_historical_manifest.json` fija sus hashes; la validación actual se guarda
por separado en `../LINEA_H_T3_V2_CIERRE_2026-10-15.md`.

Esquema actual: `filtro_*` es el lector siempre activo; `filtro_neto_*`
aplica abstención por fricción y `cobertura_neta_*` informa la participación.
NaN significa sin observaciones, no retorno cero. `ev_R_*` legado es condicional
a resolución; `bracket_ev_R_*` incluye vencidos y excluye censura. `b_real` usa
PnL en USD, por lo que mezcla geometría y tamaño. Los lectores se evalúan sobre
ventanas solapadas: sus retornos por señal no son una cartera realizable.

Unidades: `*_bps` puntos básicos; `d_pct`/`sigma_pct` porcentajes;
`pnl_neto`/`capital_final` USD; `dd_max` fracción; `rechazos` diferencia de
contadores por evaluación. Las referencias descuentan 10 pb, mientras el motor
puede cargar spread, slippage y otros costes.
